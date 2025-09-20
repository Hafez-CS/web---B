from django.urls import path
from .views import chat_room, get_ai_response, create_room, list_rooms, room_detail
from rest_framework.decorators import api_view

urlpatterns = [
    path('rooms/', list_rooms, name='room-list'),
    path('rooms/create/', create_room, name='room-create'),
    path('rooms/<int:room_id>/', room_detail, name='room-detail'),
    path('rooms/<int:room_id>/messages/', chat_room, name='room-messages'),
    path('response/', get_ai_response, name='ai-response'),
]