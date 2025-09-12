from rest_framework import serializers
from login_signup.models import User
from .models import UploadedFile



class UserSerializer(serializers.ModelSerializer):
    email = serializers.EmailField()
    username = serializers.CharField(read_only=True)

    class Meta:
        model = User
        fields = ['email' , 'first_name' ,'last_name' , 'username' , 'profile_pic_path']


    def update(self, instance, validated_data):
        if 'email' in validated_data:
            instance.email = validated_data['email']
        if 'first_name' in validated_data:
            instance.first_name = validated_data['first_name']
        if 'last_name' in validated_data:
            instance.last_name = validated_data['last_name']
        if 'profile_pic_path' in validated_data:
            instance.profile_pic_path = validated_data['profile_pic_path']
        instance.save()
        return instance


class UploadedFileSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()

    class Meta:
        model = UploadedFile
        fields = ['id' , 'file' , 'file_url' , 'uploaded_at']
        read_only_fields = ['id' , 'file_url' , 'uploaded_at']

    def get_file_url(self, obj):
        return obj.file.url if obj.file else None

    def create(self, validated_data):
        return UploadedFile.objects.create(**validated_data)