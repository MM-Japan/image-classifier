from django.contrib import admin
from django.urls import path, include  # Use include to connect app URLs

urlpatterns = [
    path('admin/', admin.site.urls),  # Admin panel
    path('', include('classify.urls')),  # Root path delegated to classify app
]
