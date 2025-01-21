# **Loan Prediction Django Project**

## Огляд
Цей проект прогнозує схвалення кредитів на основі введених даних користувачем. 
Модель машинного навчання використовує ряд характеристик заявника для визначення 
ймовірності схвалення кредиту.

## Вміст
1. [Вимоги](#Вимоги)
2. [Встановлення](#Встановлення)
3. [Структура проекту](#Структура-проекту)
4. [Функції та методи](#Функції-та-методи)
5. [Тестування](#Тестування)

## Вимоги
- Python 3.8+
- Django 3.2+
- pandas
- scikit-learn
- joblib
- plotly

## Встановлення
1. Клонуйте репозиторій:
    ```bash
    git clone https://github.com/ваш-репозиторій.git
    ```
2. Перейдіть до каталогу проекту:
    ```bash
    cd loan_prediction_project
    ```
3. Встановіть залежності:
    ```bash
    pip install -r requirements.txt
    ```
4. Виконайте міграції бази даних:
    ```bash
    python manage.py migrate
    ```
5. Запустіть сервер:
    ```bash
    python manage.py runserver
    ```


## Структура проекту

### manage.py
Основний файл для керування Django проектом. Використовується для запуску серверу,
виконання міграцій та інших команд.

### loan_prediction_project/settings.py
Конфігураційний файл проекту, що містить налаштування бази даних, додатків, середовища тощо.

### predictions/views.py
Містить функції-обробники запитів:
- `predict(request)`: Обробляє прогнозування кредиту на основі вхідних даних користувача.
- `feature_importance_view(request)`: Відображає графік важливості ознак.

### predictions/templates/predictions/
Містить HTML шаблони для відображення:

- `predict.html`: Форма для введення даних користувача.
- `result.html`: Сторінка з результатом прогнозування.
- `feature_importance.html`: Сторінка з графіком важливості ознак.

### predictions/utils/loan_prediction.py
Містить функцію predict_with_rule, яка здійснює прогнозування кредиту з урахуванням правила 
співвідношення доходу до щомісячного платежу..

### predictions/urls.py
Налаштовує маршрути для додатку `predictions`.

### train_model.py

1. Завантаження даних: Завантаження даних з CSV файлу.


  ``` data_path = r'../loan_prediction_project/predictions/data/loan_data.csv'
    data = pd.read_csv(data_path)
   ```

 
2. Імпорт необхідних бібліотек: pandas, seaborn, matplotlib, os, joblib, json, numpy, scikit-learn.
3. Обробка відсутніх значень: Видалення записів з відсутніми значеннями.

```
   data = data.dropna()
 ```
4. Додавання нових ознак: Обчислення нових ознак, таких як Income_to_Loan_Ratio, Log_ApplicantIncome, Income_Category.

   
   ```data['Income_to_Loan_Ratio'] = data['ApplicantIncome'] / data['LoanAmount']
   data['Log_ApplicantIncome'] = np.log(data['ApplicantIncome'] + 1)
   data['Income_Category'] = pd.qcut(data['ApplicantIncome'], q=3, labels=['Low', 'Medium', 'High']) 
  ```
Income_to_Loan_Ratio — співвідношення доходу заявника до суми кредиту.
Log_ApplicantIncome — логарифмований дохід для зменшення розбіжностей між великими та малими значеннями. 
Income_Category — категоризація доходу на три групи (Low, Medium, High) за допомогою qcut

 
5. Кореляційний аналіз: Аналіз кореляційних зв'язків між числовими ознаками за допомогою heatmap.


  ```corr_matrix = numeric_data.corr()
  plt.figure(figsize=(12, 8))
  sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
  plt.title('Correlation Matrix')
  plt.show() `
```


Вибираються лише числові стовпці.
Створюється кореляційна матриця, яка показує залежності між числовими ознаками.
Матриця візуалізується за допомогою теплової карти.

 
6. Обробка категоріальних змінних: Перетворення категоріальних ознак у dummy variables.

````
    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ````
Цільова змінна (Loan_Status) відокремлюється від ознак.
Категоріальні змінні кодуються в one-hot формат за допомогою pd.get_dummies(). 
Дані діляться на навчальну та тестову вибірки.


7. Попередня обробка числових даних: Створення ColumnTransformer для стандартизації числових ознак.

````
  preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train.columns)
    ],
    remainder='passthrough'`
   )
   ````
StandardScaler нормалізує числові ознаки.
ColumnTransformer застосовує цей масштабатор до всіх стовпців навчального набору.

 
8. Створення пайплайна: Пайплайн, який включає попередню обробку та модель RandomForestClassifier.

````
    pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
    ])
````

Пайплайн автоматизує обробку даних і навчання моделі.
Використовується RandomForestClassifier для класифікації.

 
9. Визначення параметрів для GridSearchCV: Параметри, які будуть оптимізовані у GridSearchCV
   Виконання GridSearchCV: Навчання моделі та оптимізація параметрів.

````
    param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_features': ['sqrt'],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [5, 10],
    'classifier__min_samples_leaf': [2, 4],
    'classifier__bootstrap': [True]
     }
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
````

param_grid визначає параметри для перебору.
GridSearchCV здійснює пошук найкращих параметрів моделі через крос-валідацію.


10. Збереження моделі: Збереження навченої моделі у файл з використанням joblib.

````
    best_pipeline = grid_search.best_estimator_
    joblib.dump(best_pipeline, model_path)
````

Зберігається найкраща модель, знайдена GridSearchCV.


11. Збереження важливості ознак: Визначення важливості ознак і збереження їх у JSON файл.

````

    feature_importances = best_pipeline.named_steps['classifier'].feature_importances_
    feature_names = X_train.columns.tolist()
    feature_importance_dict = {'features': feature_names, 'importances': feature_importances.tolist()}
    with open(feature_importance_path, 'w') as f:
    json.dump(feature_importance_dict, f)`
````

Витягується важливість ознак для пояснення моделі.Зберігається в JSON-файл.

 
12. Прогнозування на тестових даних: Виконання прогнозування на тестових даних.

````
  input_data = {
    'Loan_ID': 'NA',
    'Gender': 'Male',
    'Married': 'No',
    'Dependents': '1',
    'Education': 'Graduate',
    'Self_Employed': 'Yes',
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 0,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360,
    'Credit_History': 1.0,
    'Property_Area': 'Urban'
}
````
13. Застосування правила для оновлення прогнозів: Перевірка, чи дохід більший за щомісячний платіж, і 
    оновлення прогнозів на основі цього правила.

````
   X_test['Manual_Override'] = (X_test['Income_to_Loan_Ratio'] > 1).astype(int)
   y_pred_final = [1 if override == 1 else 0 for override, pred in zip(X_test['Manual_Override'], y_pred)]`

````
Ручне правило: якщо співвідношення доходу до кредиту більше 1, кредит схвалюється незалежно від моделі. 
 
14. Виведення фінальних результатів: Виведення оригінальних та оновлених прогнозів.

````
  result = predict_with_rule(input_data)
  print(f"Результат прогнозу: {result}")`
  ````
## Функції та методи
### Функція `predict_with_rule`


    """ 
    Прогнозує рішення про кредит з урахуванням правила співвідношення доходу до щомісячного платежу.
    Args: 
    input_data (dict): Вхідні дані для прогнозування. 
    Returns:
    str: 'Approved' або 'No Approved'. 
    """
`def predict_with_rule(input_data, model, X_train_columns):
    df = pd.DataFrame([input_data])

    df['Monthly_Payment'] = df['LoanAmount'] / df['Loan_Amount_Term']

    if 'Total_Income' not in df.columns:
        df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']

    df = pd.get_dummies(df, drop_first=True)

    missing_cols = list(set(X_train_columns) - set(df.columns))
    for col in missing_cols:
        df[col] = 0
    df = df[X_train_columns]

    if 'Total_Income' not in df.columns:
        df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    if 'Monthly_Payment' not in df.columns:
        df['Monthly_Payment'] = df['LoanAmount'] / df['Loan_Amount_Term']

    if df['Total_Income'].iloc[0] < df['Monthly_Payment'].iloc[0]:
        return 'No Approved'

    prediction = model.predict(df)[0]

    return 'Approved' if prediction == 'Y' else 'No Approved'
`
### Функція `predict`
def predict(request):
   
    """


    Обробляє POST та GET запити для прогнозування схвалення кредиту.
    
    Якщо запит метод POST, функція отримує дані з форми, створює DataFrame
    і виконує прогнозування за допомогою завантаженої моделі.
    Якщо запит метод GET, функція повертає порожню форму.
    
    Параметри:
    request (HttpRequest): HTTP запит.
    
    Повертає:
    HttpResponse: Відповідь з результатом прогнозування або форма.
    """

`def predict(request):
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
`
### Функція safe_float
def safe_float(value):
   
    """

    Конвертує значення у float з перевіркою на помилки.
    
    """
`def safe_float(value):
    try:
        return float(value)if value else 0.0
    except ValueError:
        return 0.0  # Якщо не вдається перетворити значення в float, повертаємо 0.0`

### Функція feature_importance_view
def feature_importance_view(request):

    """

    Відображає графік важливості ознак моделі.
    
    Завантажує важливість ознак з JSON файлу та створює графік
    за допомогою Plotly. Графік рендериться у HTML шаблоні.
    
    Параметри:
    request (HttpRequest): HTTP запит.
    
    Повертає:
    HttpResponse: Відповідь з графіком важливості ознак.
    """
 `def feature_importance_view(request):
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
        return JsonResponse({'error': str(e)}, status=400)`
'`''
### Тестування

Тестування проекту здійснюється за допомогою модуля Django TestCase.

Для запуску тестів використовуйте команду:
```bash
    python -m unittest discover
  ```

### Функція def setUp(self):
    # Ініціалізація клієнта для тестування HTTP-запитів
    # Путь до моделі машинного навчання
    # Перевірка існування моделі

    `def setUp(self):
        self.client = Client()

        self.model_path = '../loan_prediction_project/loan_prediction_model.pkl'

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None`

  
### Функція def test_model_loading(self):
    """Перевірка, чи завантажується модель успішно."""
    `def test_model_loading(self):
        self.assertIsNotNone(self.model, "Модель Machine Learning не найдена или не загружена")`

### Функція def test_model_prediction(self):
    """Перевірка, чи робить модель прогнози."""

    `def test_model_prediction(self):
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
            self.assertIn(prediction[0], ['Y', 'N'], "Некорректное предсказание модели")`
### Функція def test_homepage_view(self):
    """Перевірка доступності домашньої сторінки."""
    
    `def test_homepage_view(self):
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Loan Prediction")`