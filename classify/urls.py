from django.urls import path
from . import views

urlpatterns = [
    path('detect/', views.detect_objects, name='detect_objects'),  # Object detection endpoint
    path('', views.live_feed, name='live_feed'),  # Root path for the live feed page
]
