from django.shortcuts import render
from django.http import JsonResponse
import os
import joblib
import json
import plotly.graph_objects as go
from .utils.loan_prediction import predict_with_rule

MODEL_PATH = r'../loan_prediction_project/predictions/model/trained_model.pkl'
model = joblib.load(MODEL_PATH)

X_train_columns = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
                   'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']


def predict(request):
    print(f"Метод запиту: {request.method}")
    if request.method == 'POST':
        try:
            print("Отримано POST-запит")

            input_data = {
                'Loan_ID': 'NA',
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

            for key, value in input_data.items():
                if value is None or value == '':
                    return render(request, 'predictions/predict.html', {'error': 'Усі поля '
                                                                                 'повинні бути заповнені'})

            result = predict_with_rule(input_data, model, X_train_columns)
            print(f"Результат прогнозування: {result}")

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
        return float(value)if value else 0.0
    except ValueError:
        return 0.0  # Якщо не вдається перетворити значення в float, повертаємо 0.0


def feature_importance_view(request):
    print("Функція feature_importance_view викликана")
    try:
        feature_importance_path = os.path.join('model', 'feature_importance.json')
        print(f"Шлях до JSON: {feature_importance_path}")

        with open(feature_importance_path, 'r') as f:
            data = json.load(f)

        features = data['features']
        importances = data['importances']

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

        plot_html = fig.to_html(full_html=False)
        print("Графік згенеровано успішно")

        template_path = r'C:\Users\kobza\PycharmProjects\pythonProject2\loan_prediction_project\predictions\templates\predictions\feature_importance.html'
        print(f"Шлях до шаблону: {template_path}")

        return render(request, template_path, {'plot_html': plot_html})
    except Exception as e:
        print(f"Помилка: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)





