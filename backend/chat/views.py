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
        api_key=os.getenv('DEEPSEEK_API_KEY', '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'),
        base_url="https://openrouter.ai" )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

@csrf_exempt
@api_view(['GET', 'POST'])
def chat_room(request, room_id=None):
    if not request.user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    
    try:
        room = ChatRoom.objects.get(id=room_id, user=request.user)
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

            # Get chat history
            history = Message.objects.filter(room=room).order_by('timestamp')
            
            # Get AI response
            ai_response = get_ai_response(user_message, history)
            
            # Create and save the message
            message = Message.objects.create(
                room=room,
                user=request.user,  # اضافه کردن کاربر به پیام
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
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=False,
            stop=None
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