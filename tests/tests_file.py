import joblib
import os
from django.test import TestCase, Client
from django.urls import reverse
import django
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'loan_prediction_project.settings')
django.setup()
class LoanPredictionTests(TestCase):
    def setUp(self):
        self.client = Client()

        self.model_path = '../loan_prediction_project/loan_prediction_model.pkl'

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None

    def test_model_loading(self):
        self.assertIsNotNone(self.model, "Модель Machine Learning не найдена или не загружена")

    def test_model_prediction(self):
        if self.model:
            test_data = pd.DataFrame([{
                'Loan_ID': 'NA',
                'Gender': 'Male',
                'Married': 'No',
                'Dependents': '0',
                'Education': 'Graduate',
                'Self_Employed': 'No',
                'ApplicantIncome': 5000.0,
                'CoapplicantIncome': 0.0,
                'LoanAmount': 200.0,
                'Loan_Amount_Term': 360.0,
                'Credit_History': 1.0,
                'Property_Area': 'Urban'
            }])

            print(f"DataFrame для тестування: {test_data}")
            print(f"Типи даних в DataFrame:\n{test_data.dtypes}")

            prediction = self.model.predict(test_data)
            self.assertIn(prediction[0], ['Y', 'N'], "Некорректное предсказание модели")

    def test_homepage_view(self):
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Loan Prediction")





