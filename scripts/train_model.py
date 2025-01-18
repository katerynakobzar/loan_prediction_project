import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV

# Завантаження даних
data_path = r'/loan_prediction_project/predictions/data/loan_data.csv'
data = pd.read_csv(data_path)

# Обробка відсутніх значень
data = data.dropna()

# Розділення ознак і цільової змінної
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Розділення на навчальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Визначення категоріальних і числових ознак
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

# Попередня обробка даних
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Створення пайплайна
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Визначення параметрів для GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

# Виконання GridSearchCV
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Виведення найкращих параметрів
print(f"Найкращі параметри: {grid_search.best_params_}")

# Найкраща модель після GridSearchCV
best_pipeline = grid_search.best_estimator_

# Збереження моделі
model_path = os.path.join('model', 'trained_model.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(best_pipeline, model_path)

# Збереження важливості ознак
feature_importances = best_pipeline.named_steps['classifier'].feature_importances_
feature_names = X_train.columns.tolist()

feature_importance_dict = {
    'features': feature_names,
    'importances': feature_importances.tolist()
}

feature_importance_path = os.path.join('model', 'feature_importance.json')
with open(feature_importance_path, 'w') as f:
    json.dump(feature_importance_dict, f)

print("Model and feature importance saved successfully.")




