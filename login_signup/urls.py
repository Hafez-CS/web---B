from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)
from .views import UserViewSet, register, login

router = DefaultRouter()
router.register(r'users', UserViewSet, basename='user')

urlpatterns = [
    path('', include(router.urls)),
    # JWT Authentication endpoints
    path('auth/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    # Custom authentication endpoints
    path('auth/register/', register, name='register'),
    path('auth/login/', login, name='login'),
]

