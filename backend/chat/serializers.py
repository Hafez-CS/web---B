from rest_framework import serializers
from .models import Message, ChatRoom, UploadedFile

class MessageSerializer(serializers.ModelSerializer):
    user_id = serializers.IntegerField(source='user.id', read_only=True)
    room_id = serializers.IntegerField(source='room.user_room_id', read_only=True)
    
    class Meta:
        model = Message
        fields = ['id', 'user_message', 'ai_response', 'timestamp', 'room_id', 'user_id']
        read_only_fields = ['id', 'ai_response', 'timestamp', 'user_id', 'room_id']

class ChatRoomSerializer(serializers.ModelSerializer):
    user_id = serializers.IntegerField(source='user.id', read_only=True)
    name = serializers.CharField(max_length=200)
    # message_count = serializers.SerializerMethodField()
    # file_count = serializers.SerializerMethodField()
    
    class Meta:
        model = ChatRoom
        fields = ['id', 'user_room_id', 'name', 'created_at', 'user_id']
        read_only_fields = ['id', 'user_room_id', 'created_at', 'user_id']
    
    # def get_message_count(self, obj):
    #     return obj.message_set.count()
    
    # def get_file_count(self, obj):
    #     return obj.uploadedfile_set.count()

class UploadedFileSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()
    file_name = serializers.SerializerMethodField()
    user_id = serializers.IntegerField(source='user.id', read_only=True)
    room_id = serializers.IntegerField(source='room.user_room_id', read_only=True, allow_null=True)

    class Meta:
        model = UploadedFile
        fields = ['id', 'user_id', 'file', 'file_url', 'file_name', 'ai_response', 'created_at', 'room_id', 'result_json']
        read_only_fields = ['id', 'user_id', 'file_url', 'file_name', 'created_at', 'room_id']

    def get_file_url(self, obj):
        return obj.file.url if obj.file else None
    
    def get_file_name(self, obj):
        return obj.file.name if obj.file else 'Unknown'

