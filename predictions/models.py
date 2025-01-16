from django.db import models

class Prediction(models.Model):
    applicant_name = models.CharField(max_length=100)
    result = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)
