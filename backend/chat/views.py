from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import ChatRoom, Message
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view , permission_classes , parser_classes
from login_signup.models import User
from .models import UploadedFile
from .serializers import (
    UploadedFileSerializer, 
    MessageSerializer, 
    ChatRoomSerializer
)
from rest_framework.parsers import FileUploadParser , MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .ai_utils import get_ai_response
import requests
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(env_path)

@csrf_exempt
@api_view(['GET', 'POST'])
def chat_room(request, room_id=None):
    if not request.user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    
    try:
        room = ChatRoom.objects.get(user_room_id=room_id, user=request.user)
    except ChatRoom.DoesNotExist:
        return Response({'error': 'Room not found'}, status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        # دریافت پیام‌ها
        messages = Message.objects.filter(room=room).order_by('timestamp')
        
        # دریافت فایل‌های آپلود شده در این روم
        uploaded_files = UploadedFile.objects.filter(room=room).order_by('created_at')
        
        # serialize کردن پیام‌ها
        message_serializer = MessageSerializer(messages, many=True)
        message_data = message_serializer.data
        
        # serialize کردن فایل‌ها
        file_serializer = UploadedFileSerializer(uploaded_files, many=True)
        file_data = file_serializer.data
        
        # افزودن فیلد type و created_at برای مرتب‌سازی
        for item in message_data:
            item['type'] = 'message'
            item['created_at'] = item['timestamp']
        
        for item in file_data:
            item['type'] = 'file'
            # created_at از قبل موجود است
        
        # ترکیب و مرتب‌سازی بر اساس زمان
        combined_data = message_data + file_data
        combined_data.sort(key=lambda x: x['created_at'])
        
        # آماده‌سازی پاسخ نهایی
        response_data = {
            'room_id': room.user_room_id,
            'room_name': room.name,
            'timeline': combined_data
        }
        
        return Response(response_data)
    
    elif request.method == 'POST':
        try:
            user_message = request.data.get('user_message')
            if not user_message:
                return Response({'error': 'user_message is required'}, status=status.HTTP_400_BAD_REQUEST)

            # دریافت پاسخ AI با تحلیل خودکار (فقط از پیام‌های همین چت روم)
            ai_response = get_ai_response(user_message, request.user, room_id)
            
            # Create and save the message
            message = Message.objects.create(
                room=room,
                user=request.user,
                user_message=user_message,
                ai_response=ai_response
            )
            
            return Response(MessageSerializer(message).data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


# حذف تابع‌های اضافی که جایگزین شدند

@csrf_exempt
@api_view(['POST'])
def create_room(request):
    if not request.user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)

    serializer = ChatRoomSerializer(data=request.data)
    if serializer.is_valid():
        room = serializer.save(user=request.user)
        return Response(ChatRoomSerializer(room).data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@csrf_exempt
@api_view(['GET'])
def list_rooms(request):
    if not request.user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)

    rooms = ChatRoom.objects.filter(user=request.user).order_by('-created_at')
    serializer = ChatRoomSerializer(rooms, many=True)
    return Response(serializer.data)

@csrf_exempt
@api_view(['GET', 'PUT', 'DELETE'])
def room_detail(request, room_id):
    if not request.user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    
    try:
        room = ChatRoom.objects.get(user_room_id=room_id, user=request.user)
    except ChatRoom.DoesNotExist:
        return Response({'error': 'Room not found'}, status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = ChatRoomSerializer(room)
        return Response(serializer.data)
    
    elif request.method == 'PUT':
        serializer = ChatRoomSerializer(room, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        room.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FileUploadParser])
def upload_file(request, room_id):
    """
    API واحد برای آپلود فایل و تحلیل AI در یک اتاق مشخص
    - فایل آپلود می‌شود
    - بلافاصله تحلیل AI انجام می‌شود  
    - نتیجه تحلیل برگردانده می‌شود
    - room_id اجباری است
    """
    try:
        # دریافت فایل از request
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

        # دریافت target_column از request (اختیاری - اگر خالی باشد، مقدار پیش‌فرض None است)
        target_column = request.data.get('target_column', None)

        # user_id از توکن authentication گرفته می‌شود
        user_id = request.user.id
        
        # بررسی وجود room - حالا اجباری است
        try:
            room_obj = ChatRoom.objects.get(user_room_id=room_id, user=request.user)
        except ChatRoom.DoesNotExist:
            return Response({'error': 'Room not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # URL API تحلیل مالی از متغیر محیطی
        EXTERNAL_API_URL = os.getenv('FASTAPI_ANALYSIS_URL', 'http://127.0.0.1:8080/full_analysis')
        
        # آماده‌سازی داده‌ها برای ارسال
        files = {'file': (uploaded_file.name, uploaded_file.read(), uploaded_file.content_type)}
        
        # تعیین selection_mode بر اساس target_column
        selection_mode = 'ai' if target_column == None else 'user'
        target_col_value = None if target_column == None else target_column
        
        data = {
            'selection_mode': selection_mode,  # حالت انتخاب (ai یا user)
            'target_column': target_col_value,  # نام ستون هدف یا None برای AI
            'user_id': user_id,
            'room_id': room_id,
        }
        
        # ارسال درخواست به API تحلیل مالی
        try:
            response = requests.post(
                EXTERNAL_API_URL,
                files=files,
                data=data,
                timeout=300
            )
            
            if response.status_code == 200:
                api_result = response.json()
                
                # ذخیره نتیجه در دیتابیس
                try:
                    uploaded_record = UploadedFile.objects.create(
                        user=request.user,
                        file=uploaded_file,
                        room=room_obj,
                        ai_response=api_result.get('message', ''),
                        result_json=json.dumps(api_result)
                    )
                    
                    # استفاده از serializer برای پاسخ
                    file_serializer = UploadedFileSerializer(uploaded_record)
                    
                    # پاسخ موفق
                    return Response({
                        'success': True,
                        'message': 'File analyzed successfully',
                        'analysis_result': api_result,
                        'uploaded_file': file_serializer.data,
                        'file_info': {
                            'name': uploaded_file.name,
                            'size': uploaded_file.size,
                            'type': uploaded_file.content_type
                        }
                    }, status=status.HTTP_200_OK)
                    
                except Exception as db_error:
                    # اگر ذخیره در دیتابیس مشکل داشت
                    return Response({
                        'success': True,
                        'message': 'File analyzed successfully but database save failed',
                        'analysis_result': api_result,
                        'database_warning': str(db_error),
                        'file_info': {
                            'name': uploaded_file.name,
                            'size': uploaded_file.size,
                            'type': uploaded_file.content_type
                        }
                    }, status=status.HTTP_200_OK)
                    
            else:
                return Response({
                    'success': False,
                    'error': f'Analysis API error: {response.status_code}',
                    'message': response.text
                }, status=status.HTTP_400_BAD_REQUEST)
                
        except requests.RequestException as e:
            return Response({
                'success': False,
                'error': 'Failed to connect to analysis API',
                'message': str(e)
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_files(request):
    files = UploadedFile.objects.filter(user=request.user)
    serializer = UploadedFileSerializer(files, many=True)
    return Response(serializer.data)


