from django.urls import path
from .views import get_dollar, get_bors, get_gold, get_bitcoin, get_all_currencies

urlpatterns = [
    path('dollar', get_dollar, name='get_dollar'),
    path('gold', get_gold, name='get_gold'),
    path('bors', get_bors, name='get_bors'),
    path('bitcoin', get_bitcoin, name='get_bitcoin'),
    path('all', get_all_currencies, name='get_all_currencies'),
]