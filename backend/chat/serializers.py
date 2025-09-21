from rest_framework import serializers
from .models import Message, ChatRoom 
class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'user_message', 'ai_response', 'timestamp', 'room', 'user']
        read_only_fields = ['id', 'ai_response', 'timestamp', 'user']

class ChatRoomSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatRoom
        fields = ['id', 'name', 'created_at', 'user']
        read_only_fields = ['id', 'created_at', 'user']
