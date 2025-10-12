from django.db.models import Max
from .models import ChatRoom
from login_signup.models import User

def sync_room_counter(user):
    """
    همگام‌سازی room_counter کاربر با تعداد واقعی اتاق‌ها
    """
    actual_count = ChatRoom.objects.filter(user=user).count()
    if user.room_counter != actual_count:
        user.room_counter = actual_count
        user.save()