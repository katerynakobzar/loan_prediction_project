# Register your models here.
from django.contrib import admin
from .models import Prediction

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('applicant_name', 'result', 'created_at')
