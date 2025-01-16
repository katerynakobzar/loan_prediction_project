from django import forms

class LoanForm(forms.Form):
    applicant_name = forms.CharField(label='Applicant Name', max_length=100)
    gender = forms.ChoiceField(choices=[('Male', 'Male'), ('Female', 'Female')])
    married = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    dependents = forms.ChoiceField(choices=[('0', '0'), ('1', '1'), ('2', '2'), ('3+', '3+')])
    education = forms.ChoiceField(choices=[('Graduate', 'Graduate'), ('Not Graduate', 'Not Graduate')])
    self_employed = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    applicant_income = forms.FloatField(label='Applicant Income')
    coapplicant_income = forms.FloatField(label='Coapplicant Income', required=False)
    loan_amount = forms.FloatField(label='Loan Amount')
    loan_amount_term = forms.FloatField(label='Loan Amount Term')
    credit_history = forms.ChoiceField(choices=[('1.0', 'Yes'), ('0.0', 'No')])
    property_area = forms.ChoiceField(choices=[('Urban', 'Urban'), ('Semiurban', 'Semiurban'), ('Rural', 'Rural')])
