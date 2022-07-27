from django.contrib import admin
from django.urls import path
from . import views
from django.views.generic import RedirectView

urlpatterns = [
    path('classify/', views.classify, name='classify'),
    path('search/', views.search, name = 'search'),
    path('search/<str:method>', views.search, name='search'),
    path('', RedirectView.as_view(url='search')),
]
