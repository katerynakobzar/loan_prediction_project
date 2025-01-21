import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from loan_prediction_project.predictions.utils.loan_prediction import predict_with_rule

data_path = r'../loan_prediction_project/predictions/data/loan_data.csv'
data = pd.read_csv(data_path)

data = data.dropna()

data['Income_to_Loan_Ratio'] = data['ApplicantIncome'] / data['LoanAmount']
data['Log_ApplicantIncome'] = np.log(data['ApplicantIncome'] + 1)
data['Income_Category'] = pd.qcut(data['ApplicantIncome'], q=3, labels=['Low', 'Medium', 'High'])

numeric_data = data.select_dtypes(include=[np.number])

corr_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train.columns)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Визначення параметрів для GridSearchCV
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

# Виведення найкращих параметрів
print(f"Найкращі параметри: {grid_search.best_params_}")

# Найкраща модель після GridSearchCV
best_pipeline = grid_search.best_estimator_

model_path = os.path.join('model', 'trained_model.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(best_pipeline, model_path)

feature_importances = best_pipeline.named_steps['classifier'].feature_importances_
feature_names = X_train.columns.tolist()

feature_importance_dict = {
    'features': feature_names,
    'importances': feature_importances.tolist()
}

feature_importance_path = r'loan_prediction_project/predictions/model, feature_importance.json'
with open(feature_importance_path, 'w') as f:
    json.dump(feature_importance_dict, f)

print("Model and feature importance saved successfully.")

y_pred = best_pipeline.predict(X_test)

X_test['Manual_Override'] = (X_test['ApplicantIncome'] > X_test['LoanAmount'] * 1000 / 12).astype(int)
X_test['Manual_Override'] = (X_test['Income_to_Loan_Ratio'] > 1).astype(int)

y_pred_final = [
    1 if override == 1 else 0
    for override, pred in zip(X_test['Manual_Override'], y_pred)
]

print("Оригінальні прогнози:", y_pred)
print("Фінальні прогнози з урахуванням правила:", y_pred_final)

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

# Прогноз з урахуванням правила
result = predict_with_rule(input_data)
print(f"Результат прогнозу: {result}")

