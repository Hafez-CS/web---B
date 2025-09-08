from django.db import models
from django.core.validators import FileExtensionValidator
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    bio = models.CharField(blank=True , max_length=255)


class Profile(models.Model):
    user = models.OneToOneField(to=CustomUser,on_delete=models.CASCADE)
    full_name = models.CharField(max_length=100 , blank=True)
    profile_picture = models.ImageField(upload_to='profiles', blank=True , null=True)

    def __str__(self):
        return self.user.username


class UploadedFile(models.Model):
    user = models.ForeignKey(Profile, on_delete=models.CASCADE)
    file = models.FileField(upload_to='uploads/', validators=[FileExtensionValidator(allowed_extensions=['pdf', 'xlsx', 'xls' , 'jpg', 'jpeg', 'png'])])

    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.file.name}"
