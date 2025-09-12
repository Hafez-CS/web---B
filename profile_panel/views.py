from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view , permission_classes , parser_classes
from rest_framework.parsers import FileUploadParser , MultiPartParser
from rest_framework.permissions import IsAuthenticated
from login_signup.models import User
from .models import UploadedFile
from .serializers import UserSerializer , UploadedFileSerializer


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_profile(request):
    try:
        user = User.objects.get(id=request.user.id)
    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
    serializer = UserSerializer(user)
    return Response(serializer.data)


@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_profile(request):
    try:
        user = User.objects.get(id=request.user.id)
    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
    serializer = UserSerializer(user, data=request.data, partial=True)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FileUploadParser])
def upload_file(request):
    serializer = UploadedFileSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save(user=request.user)
        return Response(serializer.data)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_files(request):
    file = UploadedFile.objects.get(user=request.user)
    serializer = UploadedFileSerializer(file , many=True)
    return Response(serializer.data)