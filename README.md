# **Loan Prediction Django Project**

## Огляд
Цей проект прогнозує схвалення кредитів на основі введених даних користувачем. 
Модель машинного навчання використовує ряд характеристик заявника для визначення 
ймовірності схвалення кредиту.

## Вміст
1. [Вимоги](#Вимоги)
2. [Встановлення](#Встановлення)
3. [Структура проекту](#Структура-проекту)
4. [Опис файлів](#Опис-файлів)
5. [Функції та методи](#Функції-та-методи)
6. [Тестування](#Тестування)

## Вимоги
- Python 3.8+
- Django 3.2+
- pandas
- scikit-learn
- joblib
- plotly

## Встановлення {#Встановлення}
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


## Структура проекту {#Структура-проекту}

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

### predictions/urls.py
Налаштовує маршрути для додатку `predictions`.

### train_model.py

-`Завантаження даних: Завантаження даних з CSV файлу.`
-`Обробка відсутніх значень: Видалення записів з відсутніми значеннями.`
-`Розділення даних: Розділення даних на ознаки (X) і цільову змінну (y). Далі розділення
  на навчальний і тестовий набори.`
-`Попередня обробка: Створення обробника для числових та категоріальних ознак.`
-`Пайплайн: Створення пайплайна, який включає попередню обробку та модель RandomForestClassifier.`
-`Навчання моделі: Навчання моделі на навчальному наборі даних.`
-`Збереження моделі: Збереження навченої моделі у файл.`
-`Збереження важливості ознак: Визначення важливості ознак і збереження їх у JSON файл.`

## Функції та методи {#Функції-та-методи}

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

### Функція safe_float
def safe_float(value):
   
    """

    Конвертує значення у float з перевіркою на помилки.
    
    """

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
### Тестування {#Тестування}

Тестування проекту здійснюється за допомогою модуля Django TestCase.

Для запуску тестів використовуйте команду:
    ```bash
    python -m unittest discover
    ```

### Функція def setUp(self):
    # Ініціалізація клієнта для тестування HTTP-запитів
    # Путь до моделі машинного навчання
    # Перевірка існування моделі

### Функція def test_model_loading(self):
    """Перевірка, чи завантажується модель успішно."""

### Функція def test_model_prediction(self):
    """Перевірка, чи робить модель прогнози."""

### Функція def test_homepage_view(self):
    """Перевірка доступності домашньої сторінки."""
