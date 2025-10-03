from django.urls import path
from .views import chat_room, get_ai_response, create_room, list_rooms, room_detail, upload_file, get_files

urlpatterns = [
    path('rooms/', list_rooms, name='room-list'),
    path('rooms/create/', create_room, name='room-create'),
    path('rooms/<int:room_id>/', room_detail, name='room-detail'),
    path('rooms/<int:room_id>/messages/', chat_room, name='room-messages'),
    path('response/', get_ai_response, name='ai-response'),
    path('upload/', upload_file, name='upload_file'),
    path('files/', get_files, name='get_files'),
]