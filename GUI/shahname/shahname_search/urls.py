from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.search, name='search'),
    path('<str:method>', views.search, name='search'),
    
]
