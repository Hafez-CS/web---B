from rest_framework import serializers
from .models import Message

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'user', 'text', 'response', 'created_at', 'updated_at']
        read_only_fields = ['id', 'user', 'response', 'created_at', 'updated_at']
