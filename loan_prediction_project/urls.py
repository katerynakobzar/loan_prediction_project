from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('predictions.urls')),
    path('', lambda request: redirect('predict')),  # Перенаправлення на сторінку передбачення
    path('predict/', include('predictions.urls')),
    path('', include('predictions.urls'))
]

