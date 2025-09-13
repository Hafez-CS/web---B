from django.db import models
from django.core.validators import FileExtensionValidator
from login_signup.models import User
from django.core.exceptions import ValidationError

class UploadedFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='uploads/', validators=[FileExtensionValidator(allowed_extensions=['.xlsx', '.xls' ,'.csv'])])
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.file.name}"
