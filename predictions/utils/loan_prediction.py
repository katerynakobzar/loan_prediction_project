# loan_prediction.py

import pandas as pd

def predict_with_rule(input_data, model, X_train_columns):
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

