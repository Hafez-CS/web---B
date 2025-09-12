from django.db import models
from django.core.validators import FileExtensionValidator
from login_signup.models import User

class UploadedFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='uploads/', validators=[FileExtensionValidator(allowed_extensions=['pdf', 'xlsx', 'xls' , 'jpg', 'jpeg', 'png'])])

    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.file.name}"
