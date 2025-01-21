FROM python:3.8-slim

WORKDIR /app

# Копіюємо файли проєкту
COPY loan_prediction_project /app/loan_prediction_project
COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libpq-dev \
    && apt-get clean

RUN pip install --no-cache-dir -r /app/requirements.txt

# Перевірка, що файл існує
RUN [ -f /app/loan_prediction_project/scripts/train_model.py ] && echo "File exists" || echo "File not found"

# Запускаємо тренування моделі
RUN python /app/loan_prediction_project/scripts/train_model.py

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]


