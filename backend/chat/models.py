from django.db import models
from django.core.validators import FileExtensionValidator
from django.core.exceptions import ValidationError
from login_signup.models import User

class ChatRoom(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    user_room_id = models.PositiveIntegerField()
    name = models.CharField(max_length=255, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['user', 'user_room_id']
        ordering = ['user', 'user_room_id']
    
    def save(self, *args, **kwargs):
        if not self.user_room_id:
            # پیدا کردن حداکثر user_room_id برای این کاربر
            max_room_id = ChatRoom.objects.filter(user=self.user).aggregate(
                models.Max('user_room_id')
            )['user_room_id__max'] or 0
            self.user_room_id = max_room_id + 1
            # به‌روزرسانی room_counter کاربر
            self.user.room_counter = max_room_id + 1
            self.user.save()
        super().save(*args, **kwargs)
    
    def clean(self):
        if self.user_room_id <= 0:
            raise ValidationError("user_room_id must be greater than 0")
    
    def __str__(self):
        return f"{self.user.username} - Room {self.user_room_id}: {self.name}"

        
class Message(models.Model):
    room = models.ForeignKey(ChatRoom, on_delete=models.CASCADE, null=True, related_name='messages')
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='chat_messages')
    user_message = models.TextField()
    ai_response = models.TextField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

class UserAnalysis(models.Model):
    ANALYSIS_TYPES = [
        ('sentiment', 'User Sentiment Analysis'),
        ('tone', 'Communication Tone'),
        ('topic', 'Topic Interest Analysis'),
        ('preference', 'User Preferences'),
        ('behavior', 'User Behavior Pattern'),
        ('engagement', 'User Engagement Level'),
        ('user_needs', 'User Needs Assessment'),
        # Legacy types for backward compatibility
        ('communication_style', 'Communication Style'),
        ('expertise_level', 'Expertise Level'),
        ('response_preferences', 'Response Preferences'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='analyses')
    analysis_type = models.CharField(max_length=50, choices=ANALYSIS_TYPES)
    data = models.JSONField(default=dict)
    confidence_score = models.FloatField(default=0.0)
    last_updated = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['user', 'analysis_type']
        verbose_name = 'User Analysis'
        verbose_name_plural = 'User Analyses'



class UploadedFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # user_id
    file = models.FileField(upload_to='uploads/', validators=[FileExtensionValidator(allowed_extensions=['.xlsx', '.xls' ,'.csv'])])
    ai_response = models.TextField(null=True, blank=True)  # پاسخ AI
    created_at = models.DateTimeField(auto_now_add=True)  # uploaded_at rename شده
    room = models.ForeignKey(ChatRoom, on_delete=models.CASCADE, null=True, blank=True, related_name='uploaded_files')  # room_id
    result_json = models.TextField(null=True, blank=True)  # نتیجه JSON بصورت متن طولانی
    
    def __str__(self):
        return f"{self.user.username} - {self.file.name}"
