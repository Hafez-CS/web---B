from django.db.models.signals import post_delete
from django.dispatch import receiver
from .models import ChatRoom
from login_signup.models import User

@receiver(post_delete, sender=ChatRoom)
def decrease_room_counter(sender, instance, **kwargs):
    """
    کاهش room_counter کاربر هنگام حذف یک ChatRoom
    """
    user = instance.user
    if user.room_counter > 0:
        user.room_counter -= 1
        user.save()