from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict, name='home'),
    path('', views.predict, name='predict'),
    path('predict/', views.predict, name='predict'),
    path('feature-importance/', views.feature_importance_view, name='feature_importance'),
]

