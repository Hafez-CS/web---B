from rest_framework import serializers
from .models import Message, ChatRoom 
from .models import UploadedFile
class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'user_message', 'ai_response', 'timestamp', 'room', 'user']
        read_only_fields = ['id', 'ai_response', 'timestamp', 'user']

class ChatRoomSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatRoom
        fields = ['id', 'user_room_id', 'name', 'created_at', 'user']
        read_only_fields = ['id', 'user_room_id', 'created_at', 'user']

class UploadedFileSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()
    user_id = serializers.IntegerField(source='user.id', read_only=True)  # نمایش user_id
    room_id = serializers.IntegerField(source='room.user_room_id', read_only=True, allow_null=True)  # نمایش room_id

    class Meta:
        model = UploadedFile
        fields = ['id', 'user_id', 'file', 'file_url', 'ai_response', 'created_at', 'room_id', 'result_json']
        read_only_fields = ['id', 'user_id', 'file_url', 'created_at', 'room_id']

    def get_file_url(self, obj):
        return obj.file.url if obj.file else None

    def create(self, validated_data):
        return UploadedFile.objects.create(**validated_data)