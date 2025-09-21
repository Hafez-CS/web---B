from django.db import models
from login_signup.models import User

class ChatRoom(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True)

class Message(models.Model):
    room = models.ForeignKey(ChatRoom, on_delete=models.CASCADE, null=True, related_name='messages')
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='chat_messages')
    user_message = models.TextField()
    ai_response = models.TextField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']