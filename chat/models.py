from django.db import models
from django.contrib.auth.models import User

class Message(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='messages')
	text = models.TextField()
	response = models.TextField(blank=True, null=True)
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	def __str__(self):
		return f"{self.user.username}: {self.text[:30]}"
