# EasyHandWritten_app/urls.py
from django.urls import path

from EasyHandWritten_app import views

urlpatterns = [
    path('api/upload/', views.upload_image, name='upload_image'),
    path('api/history/', views.get_latest_image, name='get_latest_image'),
    path('api/history/<int:id>/', views.delete_history, name='delete_history'),
    path('api/judge_name/<int:judge_id>/', views.get_AIname , name='get_AIname'),
    path('api/getimage/<int:image_id>/', views.get_image_by_id, name='get_image_by_id'),
]
