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
from .ai_file_analyze import financial_predictor
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
    try:
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

        target_column = request.data.get('target_column', None)
        if target_column == '' or target_column == 'null':
            target_column = None
        
        try:
            room_obj = ChatRoom.objects.get(user_room_id=room_id, user=request.user)
        except ChatRoom.DoesNotExist:
            return Response({'error': 'Room not found'}, status=status.HTTP_404_NOT_FOUND)
        
        try:
            analysis_result = financial_predictor(uploaded_file, target_column)
            
            if 'error' in analysis_result:
                return Response({
                    'success': False,
                    'error': 'Analysis failed',
                    'message': analysis_result['error']
                }, status=status.HTTP_400_BAD_REQUEST)
            
            try:
                uploaded_record = UploadedFile.objects.create(
                    user=request.user,
                    file=uploaded_file,
                    room=room_obj,
                    message=analysis_result.get('message', ''),
                    target_column=analysis_result.get('target_column', ''),
                    selected_model=analysis_result.get('selected_model', ''),
                    gemini_recommendation=analysis_result.get('gemini_recommendation', ''),
                    prediction_data=json.dumps(analysis_result.get('prediction_data', {}))
                )
                
                file_serializer = UploadedFileSerializer(uploaded_record)
                
                return Response({
                    'success': True,
                    'analysis_result': analysis_result,
                    'file_user_info': file_serializer.data
                }, status=status.HTTP_200_OK)
                
            except Exception as db_error:
                return Response({
                    'success': True,
                    'message': 'File analyzed successfully but database save failed',
                    'analysis_result': analysis_result,
                    'database_warning': str(db_error),
                    'file_info': {
                        'name': uploaded_file.name,
                        'size': uploaded_file.size,
                        'type': uploaded_file.content_type
                    }
                }, status=status.HTTP_200_OK)
                
        except Exception as analysis_error:
            return Response({
                'success': False,
                'error': 'Analysis process failed',
                'message': str(analysis_error)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_files(request):
    files = UploadedFile.objects.filter(user=request.user)
    serializer = UploadedFileSerializer(files, many=True)
    return Response(serializer.data)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_room(request , room_id):
    try:
        room = ChatRoom.objects.get(user_room_id=room_id , user=request.user)
        room.delete()
        return Response({"message": "اتاق با موقیت حذف شد"}, status=status.HTTP_204_NO_CONTENT)
    except ChatRoom.DoesNotExist:
        return Response({"error": "اتاق پیدا نشد یا شما مجاز به حذف آن نیستید"}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)