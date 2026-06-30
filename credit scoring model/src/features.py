import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def engineer_features(df):
    df = df.copy()

    income_safe = df['income'].replace(0, 1)
    df['debt_to_income_ratio'] = (df['total_debt'] / income_safe).clip(0, 10)

    df['loan_to_income_ratio'] = (df['loan_amount'] / income_safe).clip(0, 5)

    df['payment_history_score'] = (1 - np.tanh(df['late_payments'] / 5)) * 100

    df['credit_account_age'] = df['credit_history_length'].clip(0, 50)

    df['default_ratio'] = (
        df['num_defaults'] / df['num_credit_accounts'].replace(0, 1)
    ).clip(0, 1)

    df['employment_stability'] = (df['employment_length'] / np.clip(df['age'] - 18, 1, 60)).clip(0, 1)

    df['financial_stress_index'] = (
        0.4 * df['debt_to_income_ratio']
        + 0.3 * df['loan_to_income_ratio']
        + 0.3 * (df['total_debt'] / income_safe)
    ).clip(0, 10)

    df['credit_utilization_score'] = (1 - df['credit_card_utilization']) * 100

    df['dependents_per_income'] = (df['dependents'] / income_safe * 100000).clip(0, 20)

    df['is_high_education'] = df['education_level'].isin(['Bachelor', 'Master', 'PhD']).astype(int)

    df['is_homeowner'] = df['home_ownership'].isin(['Own', 'Mortgage']).astype(int)

    return df


def prepare_features(df, target_col='credit_risk', scale=True):
    df = engineer_features(df)

    cat_cols = ['home_ownership', 'education_level', 'marital_status']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    if target_col in df.columns:
        y = (df[target_col].map({'Good': 0, 'Bad': 1}) if df[target_col].dtype == 'object'
             else df[target_col])
        X = df.drop(columns=[target_col])
    else:
        X = df
        y = None

    X = X.select_dtypes(include=[np.number])

    if scale:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        return X_scaled, y, scaler

    return X, y, None


FEATURE_GROUPS = {
    'Income & Debt': ['income', 'total_debt', 'debt_to_income_ratio', 'financial_stress_index'],
    'Loan Profile': ['loan_amount', 'loan_term', 'loan_to_income_ratio', 'annual_interest_rate'],
    'Payment History': ['late_payments', 'num_defaults', 'payment_history_score', 'default_ratio', 'prior_loan_defaults'],
    'Credit Profile': ['credit_history_length', 'num_credit_accounts', 'credit_account_age', 'has_credit_card', 'credit_card_utilization', 'credit_utilization_score'],
    'Demographics': ['age', 'employment_length', 'employment_stability', 'dependents', 'dependents_per_income', 'is_high_education', 'is_homeowner', 'home_ownership', 'education_level', 'marital_status'],
}
