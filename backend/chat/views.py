from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import ChatRoom, Message
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view , permission_classes , parser_classes
from login_signup.models import User
from .models import UploadedFile
from .serializers import UploadedFileSerializer
from rest_framework.parsers import FileUploadParser , MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .serializers import MessageSerializer, ChatRoomSerializer
from .ai_utils import get_ai_response
import requests
import json

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
        messages = Message.objects.filter(room=room).order_by('timestamp')
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)
    
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
def upload_file(request):
    try:
        # دریافت فایل از request
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        # دریافت room_id از request (اختیاری)
        room_id = request.data.get('room_id', None)
        
        # ارسال فایل به API خارجی
        external_api_response = send_file_to_external_api(
            uploaded_file=uploaded_file,
            user=request.user,
            room_id=room_id
        )
        
        return Response(external_api_response, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

def send_file_to_external_api(uploaded_file, user, room_id=None):
    """ارسال فایل به API تحلیل مالی"""
    import requests
    
    # URL API تحلیل مالی
    EXTERNAL_API_URL = "http://127.0.0.1:8000/full_analysis"  # آدرس FastAPI شما
    
    # آماده‌سازی داده‌ها برای ارسال
    files = {'file': (uploaded_file.name, uploaded_file.read(), uploaded_file.content_type)}
    
    # تنظیمات پیش‌فرض برای تحلیل
    config = {
        "test_size": 0.2,
        "random_state": 42,
        "max_hist": 6,
        "max_rows_preview": 5
    }
    
    data = {
        'config': json.dumps(config),  # JSON string مطابق انتظار API
        'selection_mode': 'ai',  # حالت انتخاب خودکار با AI
        'target_column': None,  # چون از حالت AI استفاده می‌کنیم
        # فیلدهای اضافی برای tracking
        'user_id': user.id,
        'room_id': room_id,
    }
    
    try:
        # ارسال درخواست به API تحلیل مالی
        response = requests.post(
            EXTERNAL_API_URL,
            files=files,
            data=data,
            timeout=300  # 60 ثانیه timeout چون تحلیل ممکن است زمان‌بر باشد
        )
        
        if response.status_code == 200:
            api_result = response.json()
            
            # ذخیره نتیجه در دیتابیس برای بازیابی بعدی
            try:
                # دریافت room object اگر room_id ارسال شده
                room_obj = None
                if room_id:
                    room_obj = ChatRoom.objects.get(user_room_id=room_id, user=user)
                
                # ایجاد رکورد در UploadedFile
                uploaded_record = UploadedFile.objects.create(
                    user=user,
                    file=uploaded_file,  # فایل را هم ذخیره می‌کنیم
                    room=room_obj,
                    ai_response=api_result.get('message', ''),
                    result_json=json.dumps(api_result)  # تمام نتیجه را ذخیره می‌کنیم
                )
                
                return {
                    'success': True,
                    'message': 'File analyzed successfully',
                    'analysis_result': api_result,
                    'database_id': uploaded_record.id,
                    'file_info': {
                        'name': uploaded_file.name,
                        'size': uploaded_file.size,
                        'type': uploaded_file.content_type
                    }
                }
                
            except Exception as db_error:
                # اگر ذخیره در دیتابیس مشکل داشت، حداقل نتیجه تحلیل را برگردان
                return {
                    'success': True,
                    'message': 'File analyzed successfully but database save failed',
                    'analysis_result': api_result,
                    'database_error': str(db_error),
                    'file_info': {
                        'name': uploaded_file.name,
                        'size': uploaded_file.size,
                        'type': uploaded_file.content_type
                    }
                }
        else:
            return {
                'success': False,
                'error': f'Analysis API error: {response.status_code}',
                'message': response.text
            }
            
    except requests.RequestException as e:
        return {
            'success': False,
            'error': 'Failed to connect to analysis API',
            'message': str(e)
        }

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_files(request):
    files = UploadedFile.objects.filter(user=request.user)
    serializer = UploadedFileSerializer(files, many=True)
    return Response(serializer.data)


