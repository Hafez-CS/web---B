from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import ChatRoom, Message
from openai import OpenAI
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import MessageSerializer, ChatRoomSerializer
import os

# تنظیمات OpenAI client
try:
    client = OpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY', 'sk-7f1b2ace00344041984102367b48983c'),
        base_url="https://api.deepseek.com/v1"  # تغییر نسخه API به v1
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

@csrf_exempt
@api_view(['GET', 'POST'])
@login_required
def chat_room(request, room_id=None):
    # First, verify the room exists and belongs to the user
    try:
        room = ChatRoom.objects.get(id=room_id, user=request.user)
    except ChatRoom.DoesNotExist:
        return Response({'error': 'Room not found'}, status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        messages = Message.objects.filter(room=room).order_by('timestamp')
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        # Create a new message
        try:
            user_message = request.data.get('user_message')
            if not user_message:
                return Response({'error': 'user_message is required'}, status=status.HTTP_400_BAD_REQUEST)

            # Get chat history
            history = Message.objects.filter(room=room).order_by('timestamp')
            
            # Get AI response
            ai_response = get_ai_response(user_message, history)
            
            # Create and save the message
            message = Message.objects.create(
                room=room,
                user_message=user_message,
                ai_response=ai_response
            )
            
            return Response(MessageSerializer(message).data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


def get_ai_response(user_message, history):
    try:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for msg in history:
            messages.append({"role": "user", "content": msg.user_message})
            if msg.ai_response and not msg.ai_response.startswith("Error in AI response"):
                messages.append({"role": "assistant", "content": msg.ai_response})
        messages.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            model="deepseek-chat",  # یا از مدل‌های دیگر استفاده کنید
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        if hasattr(response, 'choices') and len(response.choices) > 0 and hasattr(response.choices[0], 'message'):
            return response.choices[0].message.content
        else:
            return "I apologize, but I couldn't generate a response at the moment."
            
    except Exception as e:
        print(f"Deepseek API Error: {str(e)}")  # برای دیباگ
        # اینجا می‌توانید از یک سرویس جایگزین استفاده کنید
        # یا یک پاسخ پیش‌فرض برگردانید
        return "I apologize, but I'm having trouble connecting to the AI service. Please try again later."

@csrf_exempt
@api_view(['POST'])
@login_required
def create_room(request):
    """Create a new chat room"""
    serializer = ChatRoomSerializer(data=request.data)
    if serializer.is_valid():
        room = serializer.save(user=request.user)
        return Response(ChatRoomSerializer(room).data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@csrf_exempt
@api_view(['GET'])
@login_required
def list_rooms(request):
    """List all chat rooms for the current user"""
    rooms = ChatRoom.objects.filter(user=request.user).order_by('-created_at')
    serializer = ChatRoomSerializer(rooms, many=True)
    return Response(serializer.data)

@csrf_exempt
@api_view(['GET', 'PUT', 'DELETE'])
@login_required
def room_detail(request, room_id):
    """Get, update or delete a chat room"""
    try:
        room = ChatRoom.objects.get(id=room_id, user=request.user)
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