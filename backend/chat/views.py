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
    serializer = UploadedFileSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save(user=request.user)
        return Response(serializer.data)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_files(request):
    files = UploadedFile.objects.filter(user=request.user)
    serializer = UploadedFileSerializer(files, many=True)
    return Response(serializer.data)


