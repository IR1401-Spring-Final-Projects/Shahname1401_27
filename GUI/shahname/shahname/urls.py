from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('shahname_search.urls')),
    path('admin/', admin.site.urls),
]
