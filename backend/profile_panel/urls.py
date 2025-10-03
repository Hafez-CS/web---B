from django.urls import path
from .views import get_profile , update_profile

urlpatterns = [
    path('profile/', get_profile, name='get_profile'),
    path('profile/update/', update_profile, name='update_profile'),
]