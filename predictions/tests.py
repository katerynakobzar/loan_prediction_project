# tests.py у кореневій директорії
from django.test import TestCase, Client
from django.urls import reverse
from loan_prediction_project.predictions.models import Prediction  # Імпорт з додатка predictions

class LoanPredictionTests(TestCase):
    def setUp(self):
        # Ініціалізація клієнта для тестування HTTP-запитів
        self.client = Client()

        # Шлях до моделі Machine Learning
        self.model_path = r'../loan_prediction_model.pkl'

        # Перевірка існування моделі
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None

    def test_model_loading(self):
        """Перевірка, чи модель успішно завантажується."""
        self.assertIsNotNone(self.model, "Модель Machine Learning не знайдена або не завантажена")

    def test_model_prediction(self):
        """Перевірка, чи модель робить передбачення."""
        if self.model:
            test_data = [[5000, 0, 1, 3, 1, 0, 2]]  # Зразок тестових даних
            prediction = self.model.predict(test_data)
            self.assertIn(prediction[0], ['Y', 'N'], "Невірне передбачення моделі")

    def test_homepage_view(self):
        """Перевірка доступності домашньої сторінки."""
        response = self.client.get(reverse('home'))  # Замініть 'home' на назву вашого URL-нейму для домашньої сторінки
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Loan Prediction")  # Перевіряє, чи є ключовий текст на сторінці

    def test_prediction_view(self):
        """Перевірка сторінки передбачення."""
        response = self.client.post(reverse('predict'), {
            'income': 5000,
            'loan_amount': 200,
            'credit_history': 1,
            'education': 'Graduate',
            'gender': 'Male',
            'married': 'No',
            'dependents': 0
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Prediction Result")  # Перевіряє, чи є ключовий текст на сторінці результату
