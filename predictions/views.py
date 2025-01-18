from django.shortcuts import render
from django.http import JsonResponse
import os
import joblib
import pandas as pd
import json
import plotly.graph_objects as go

# Шлях до моделі
MODEL_PATH = r'../loan_prediction_project/predictions/model/trained_model.pkl'
model = joblib.load(MODEL_PATH)

def predict(request):
    print(f"Метод запиту: {request.method}")
    if request.method == 'POST':
        try:
            print("Отримано POST-запит")

            # Отримання значень з POST-запиту та перевірка на коректність
            input_data = {
                'Loan_ID': 'NA',  # Додаємо колонку Loan_ID з нульовим значенням
                'Gender': request.POST.get('Gender'),
                'Married': request.POST.get('Married'),
                'Dependents': request.POST.get('Dependents'),
                'Education': request.POST.get('Education'),
                'Self_Employed': request.POST.get('Self_Employed'),
                'ApplicantIncome': safe_float(request.POST.get('ApplicantIncome')),
                'CoapplicantIncome': safe_float(request.POST.get('CoapplicantIncome')),
                'LoanAmount': safe_float(request.POST.get('LoanAmount')),
                'Loan_Amount_Term': safe_float(request.POST.get('Loan_Amount_Term')),
                'Credit_History': safe_float(request.POST.get('Credit_History')),
                'Property_Area': request.POST.get('Property_Area')
            }
            print(f"Вхідні дані: {input_data}")
            # Перевірка, чи всі значення заповнені
            for key, value in input_data.items():
                if value is None or value == '':
                 return render(request, 'predictions/predict.html',{'error': 'Усі поля повинні бути заповнені'})
            # Створення DataFrame
            input_df = pd.DataFrame([input_data])
            print(f"DataFrame для прогнозування: {input_df}")
            print(f"Типи даних в DataFrame:\n{input_df.dtypes}")
            # Передбачення
            prediction = model.predict(input_df)[0]
            result = 'Approved' if prediction == 'Y' else 'Not Approved'
            print(f"Результат прогнозування: {result}")

            # Відображення результату
            return render(request, 'predictions/result.html', {'result': result})
        except Exception as e:
            print(f"Помилка: {str(e)}")
            return JsonResponse({'error': str(e)}, status=400)
    elif request.method == 'GET':
        print("Отримано GET-запит")
        return render(request, 'predictions/predict.html')
    print("Метод запиту не підтримується")
    return JsonResponse({'error': 'Invalid request method'}, status=405)


def safe_float(value):
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0  # Якщо не вдається перетворити значення в float, повертаємо 0.0


def feature_importance_view(request):
    print("Функція feature_importance_view викликана")
    try:
        # Шлях до JSON з важливістю ознак
        feature_importance_path = os.path.join('model', 'feature_importance.json')
        print(f"Шлях до JSON: {feature_importance_path}")

        with open(feature_importance_path, 'r') as f:
            data = json.load(f)

        features = data['features']
        importances = data['importances']

        # Створення графіка
        fig = go.Figure(
            go.Bar(
                x=features,
                y=importances,
                marker=dict(color='blue'),
            )
        )
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Features',
            yaxis_title='Importance',
            template='plotly_white'
        )

        # Генерація HTML
        plot_html = fig.to_html(full_html=False)
        print("Графік згенеровано успішно")

        # Абсолютний шлях до шаблону
        template_path = r'C:\Users\kobza\PycharmProjects\pythonProject2\loan_prediction_project\predictions\templates\predictions\feature_importance.html'
        print(f"Шлях до шаблону: {template_path}")

        return render(request, template_path, {'plot_html': plot_html})
    except Exception as e:
        print(f"Помилка: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)


