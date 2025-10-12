from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    profile_pic_path = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
    room_counter = models.PositiveIntegerField(default=0)  
    
    def __str__(self):
        return self.username
