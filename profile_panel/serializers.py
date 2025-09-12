from rest_framework import serializers
from login_signup.models import User
from .models import UploadedFile



class UserSerializer(serializers.ModelSerializer):
    email = serializers.EmailField(source='user.email')
    username = serializers.CharField(source='user.username' , read_only=True)

    class Meta:
        model = User
        fields = ['email' , 'first_name' ,'last_name' , 'username' , 'profile_pic_path']


    def update(self, instance, validated_data):
        user_data = validated_data.pop('user', {})
        user = instance.user
        if 'email' in user_data:
            user.email = user_data['email']
            user.save()
            instance = super().update(instance, validated_data)
            return instance


class UploadedFileSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()

    class Meta:
        model = UploadedFile
        fields = ['id' , 'file' , 'file_url' , 'uploaded_at']
        read_only_fields = ['id' , 'file_url' , 'uploaded_at']

    def get_file_url(self, obj):
        return obj.file_url if obj.file else None

    def create(self, validated_data):
        return UploadedFile.objects.create(**validated_data)