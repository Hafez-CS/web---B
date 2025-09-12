from django.urls import path
from .views import get_profile , update_profile , upload_file, get_files

urlpatterns = [
    path('profile/', get_profile, name='get_profile'),
    path('profile/update/', update_profile, name='update_profile'),
    path('dashboard/upload/', upload_file, name='upload_file'),
    path('dashboard/files/', get_files, name='get_files'),
]