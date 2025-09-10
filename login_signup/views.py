from django.shortcuts import render

# Create your views here.
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
from django.contrib.auth import authenticate
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import UserSerializer

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    """
    Register a new user and return JWT tokens
    """
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        user.set_password(serializer.validated_data['password'])
        user.save()
        
        refresh = RefreshToken.for_user(user)
        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'user': UserSerializer(user).data
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    """
    Login user and return JWT tokens
    """
    username = request.data.get('username')
    password = request.data.get('password')
    
    if username and password:
        user = authenticate(username=username, password=password)
        if user:
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
                'user': UserSerializer(user).data
            })
        else:
            return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
    return Response({'error': 'Username and password required'}, status=status.HTTP_400_BAD_REQUEST)
