from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import ChatRoom, Message
from openai import OpenAI

client = OpenAI(
    api_key="sk-7f1b2ace00344041984102367b48983c",  # API key رایگان DeepSeek
    base_url="https://api.deepseek.com/V3.1"  # endpoint DeepSeek
)

@login_required
def chat_room(request, room_id=None):
    if room_id:
        room = ChatRoom.objects.get(id=room_id, user=request.user)
    else:
        room = ChatRoom.objects.create(user=request.user)
        return redirect('chat_room', room_id=room.id)
    
    messages = Message.objects.filter(room=room).order_by('timestamp')
    return render(request, 'chat.html', {'room': room, 'messages': messages})

def get_ai_response(user_message, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in history:
        messages.append({"role": "user", "content": msg.user_message})
        if msg.ai_response:
            messages.append({"role": "assistant", "content": msg.ai_response})
    messages.append({"role": "user", "content": user_message})
    
    response = client.chat.completions.create(
        model="deepseek-chat",  # یا "deepseek-reasoner" برای تحلیل عمیق
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content