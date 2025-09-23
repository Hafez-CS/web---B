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