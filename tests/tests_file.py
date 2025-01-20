# tests_file.py у кореневій директорії

import joblib
import os  # Для работы с операционной системой
from django.test import TestCase, Client  # Для тестирования в Django
from django.urls import reverse  # Для получения URL по имени маршрута
import django
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'loan_prediction_project.settings')
django.setup()
class LoanPredictionTests(TestCase):
    def setUp(self):
        # Ініціалізація клієнта для тестування HTTP-запитів
        self.client = Client()

        # Путь до моделі Machine Learning
        self.model_path = '../loan_prediction_project/loan_prediction_model.pkl'

        # Перевірка існування моделі
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None

    def test_model_loading(self):
        """Проверка, загружается ли модель успешно."""
        self.assertIsNotNone(self.model, "Модель Machine Learning не найдена или не загружена")

    def test_model_prediction(self):
        """Проверка, делает ли модель предсказания."""
        if self.model:
            # Пример тестовых данных с 11 признаками
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

            # Отримання предсказання
            prediction = self.model.predict(test_data)
            self.assertIn(prediction[0], ['Y', 'N'], "Некорректное предсказание модели")

    def test_homepage_view(self):
        """Проверка доступности домашней страницы."""
        response = self.client.get(reverse('home'))  # Замените 'home' на имя вашего URL для домашней страницы
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Loan Prediction")  # Проверяет наличие ключевого текста на странице





