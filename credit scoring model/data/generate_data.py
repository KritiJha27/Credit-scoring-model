import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_credit_data(n_samples=10000, random_state=RANDOM_SEED, target_bad_rate=0.25):
    rng = np.random.RandomState(random_state)

    n_good = int(n_samples * (1 - target_bad_rate))
    n_bad = n_samples - n_good

    good = _generate_class(n_good, is_bad=False, rng=rng)
    bad = _generate_class(n_bad, is_bad=True, rng=rng)

    df = pd.concat([good, bad], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def _generate_class(n, is_bad=False, rng=None):
    if rng is None:
        rng = np.random.RandomState()

    if is_bad:
        age = rng.randint(18, 65, n).astype(float)
        income = rng.lognormal(mean=9.8, sigma=0.5, size=n).astype(float)
        income = np.clip(income, 15000, 150000)
        loan_amount = rng.lognormal(mean=10.0, sigma=0.7, size=n).astype(float)
        loan_amount = np.clip(loan_amount, 5000, 100000)
        loan_term = rng.choice([36, 48, 60, 72, 84], n, p=[0.15, 0.25, 0.35, 0.15, 0.10])
        credit_history_length = rng.randint(1, 12, n).astype(float)
        num_credit_accounts = rng.randint(1, 10, n).astype(float)
        late_payments = rng.poisson(lam=4, size=n).astype(float)
        num_defaults = rng.poisson(lam=2, size=n).astype(float)
        total_debt = rng.lognormal(mean=10.0, sigma=0.7, size=n).astype(float)
        total_debt = np.clip(total_debt, 5000, 200000)
        employment_length = rng.randint(0, 15, n).astype(float)
        employment_length = np.clip(employment_length, 0, age - 18)
        home_ownership = rng.choice(['Rent', 'Mortgage', 'Own', 'Other'], n, p=[0.50, 0.30, 0.10, 0.10])
        education_level = rng.choice(['High School', 'Associate', 'Bachelor', 'Master', 'PhD'], n, p=[0.40, 0.25, 0.25, 0.08, 0.02])
        marital_status = rng.choice(['Single', 'Married', 'Divorced', 'Widowed'], n, p=[0.45, 0.30, 0.20, 0.05])
        dependents = rng.choice([0, 1, 2, 3, 4], n, p=[0.30, 0.30, 0.25, 0.10, 0.05]).astype(float)
        has_credit_card = rng.choice([0, 1], n, p=[0.30, 0.70]).astype(float)
        credit_card_utilization = rng.beta(a=5, b=3, size=n).astype(float)
        annual_interest_rate = rng.uniform(10.0, 28.0, n).astype(float)
        prior_loan_defaults = rng.choice([0, 1], n, p=[0.40, 0.60]).astype(float)
    else:
        age = rng.randint(22, 75, n).astype(float)
        income = rng.lognormal(mean=10.8, sigma=0.5, size=n).astype(float)
        income = np.clip(income, 20000, 250000)
        loan_amount = rng.lognormal(mean=9.2, sigma=0.7, size=n).astype(float)
        loan_amount = np.clip(loan_amount, 1000, 60000)
        loan_term = rng.choice([12, 24, 36, 48, 60, 72, 84], n, p=[0.10, 0.20, 0.30, 0.20, 0.10, 0.05, 0.05])
        credit_history_length = rng.randint(5, 30, n).astype(float)
        num_credit_accounts = rng.randint(3, 20, n).astype(float)
        late_payments = rng.poisson(lam=0.5, size=n).astype(float)
        num_defaults = rng.poisson(lam=0.1, size=n).astype(float)
        total_debt = rng.lognormal(mean=8.5, sigma=0.8, size=n).astype(float)
        total_debt = np.clip(total_debt, 0, 100000)
        employment_length = rng.randint(3, 40, n).astype(float)
        employment_length = np.clip(employment_length, 0, age - 18)
        home_ownership = rng.choice(['Rent', 'Mortgage', 'Own', 'Other'], n, p=[0.25, 0.45, 0.25, 0.05])
        education_level = rng.choice(['High School', 'Associate', 'Bachelor', 'Master', 'PhD'], n, p=[0.15, 0.18, 0.40, 0.20, 0.07])
        marital_status = rng.choice(['Single', 'Married', 'Divorced', 'Widowed'], n, p=[0.28, 0.55, 0.12, 0.05])
        dependents = rng.choice([0, 1, 2, 3, 4], n, p=[0.38, 0.30, 0.18, 0.10, 0.04]).astype(float)
        has_credit_card = rng.choice([0, 1], n, p=[0.08, 0.92]).astype(float)
        credit_card_utilization = rng.beta(a=2, b=8, size=n).astype(float)
        annual_interest_rate = rng.uniform(2.0, 15.0, n).astype(float)
        prior_loan_defaults = rng.choice([0, 1], n, p=[0.92, 0.08]).astype(float)

    df = pd.DataFrame({
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'credit_history_length': credit_history_length,
        'num_credit_accounts': num_credit_accounts,
        'late_payments': late_payments,
        'num_defaults': num_defaults,
        'total_debt': total_debt,
        'employment_length': employment_length,
        'home_ownership': home_ownership,
        'education_level': education_level,
        'marital_status': marital_status,
        'dependents': dependents,
        'has_credit_card': has_credit_card,
        'credit_card_utilization': credit_card_utilization,
        'annual_interest_rate': annual_interest_rate,
        'prior_loan_defaults': prior_loan_defaults,
        'credit_risk': 'Bad' if is_bad else 'Good',
    })

    noise_level = 0.15
    swap_mask = rng.random(n) < noise_level
    df.loc[swap_mask, 'credit_risk'] = 'Bad' if not is_bad else 'Good'

    return df


if __name__ == '__main__':
    df = generate_credit_data(10000)
    df.to_csv('credit_risk_dataset.csv', index=False)
    print(f"Generated {len(df)} samples")
    print(f"Class distribution:\n{df['credit_risk'].value_counts()}")
    print(f"Columns: {list(df.columns)}")
