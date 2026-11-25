"""Generates a synthetic insurance-customer churn dataset for ModelMate.

Outputs (next to this script):
  insurance_churn_train.csv           3,000 customers incl. the `churn` target (Yes/No)
  insurance_churn_new_customers.csv   300 unseen customers WITHOUT `churn` -> upload on the Models screen
  insurance_churn_new_customers_truth.csv   true labels for those 300 (to sanity-check predictions)

Churn is driven by: short tenure, late payments, complaints, many service calls, manual payment,
no auto-renew, high premium relative to income, low digital engagement and recent rejected claims.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent


def make(n: int, seed: int, id_start: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 80, n)
    income = np.round(rng.lognormal(10.9, 0.45, n), -2)
    policy_type = rng.choice(["Auto", "Home", "Life", "Health", "Travel"], n, p=[.35, .25, .2, .15, .05])
    base_premium = {"Auto": 1100, "Home": 1400, "Life": 900, "Health": 2200, "Travel": 300}
    premium = np.round([base_premium[p] * rng.uniform(.6, 1.8) for p in policy_type], 2)
    tenure = np.round(rng.gamma(2.0, 2.5, n), 1)
    late = rng.poisson(0.8, n)
    calls = rng.poisson(1.5, n)
    complaints = rng.binomial(3, 0.08, n)
    claims = rng.poisson(0.6, n)
    claim_amt = np.where(claims > 0, np.round(claims * rng.lognormal(7.5, 0.8, n), 2), 0.0)
    last_claim = np.where(claims > 0, rng.choice(["Approved", "Rejected", "Pending"], n, p=[.7, .2, .1]), "No claim")
    pay = rng.choice(["Auto-debit", "Credit card", "Bank transfer", "Cheque"], n, p=[.4, .3, .2, .1])
    auto_renew = rng.choice(["Yes", "No"], n, p=[.6, .4])
    engagement = np.clip(np.round(rng.normal(55, 20, n) - (age - 45) * 0.4, 1), 0, 100)
    channel = rng.choice(["Agent", "Online", "Broker", "Call centre"], n, p=[.35, .35, .2, .1])
    region = rng.choice(["North", "South", "East", "West", "Central"], n)
    gender = rng.choice(["Female", "Male"], n)
    marital = rng.choice(["Single", "Married", "Divorced", "Widowed"], n, p=[.35, .45, .15, .05])
    num_policies = 1 + rng.poisson(0.7, n)
    last_contact = pd.Timestamp("2026-10-31") - pd.to_timedelta(rng.integers(0, 720, n), unit="D")

    logit = (
        -1.6 - 0.22 * np.minimum(tenure, 10) + 0.45 * late + 0.25 * calls + 0.8 * complaints
        + 0.9 * (auto_renew == "No") + 0.5 * np.isin(pay, ["Cheque", "Bank transfer"])
        + 3.0 * (premium / income * 10) - 0.02 * engagement + 1.0 * (last_claim == "Rejected")
        - 0.35 * (num_policies - 1) + 0.3 * (channel == "Online") + 0.01 * (45 - age).clip(0)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame({
        "customer_id": [f"C{id_start + i:06d}" for i in range(n)],
        "age": age, "gender": gender, "marital_status": marital, "region": region,
        "annual_income": income, "policy_type": policy_type, "num_policies": num_policies,
        "policy_tenure_years": tenure, "annual_premium": premium, "payment_method": pay,
        "auto_renew": auto_renew, "late_payments_12m": late, "num_claims_3y": claims,
        "total_claim_amount_3y": claim_amt, "last_claim_status": last_claim,
        "service_calls_12m": calls, "complaints_12m": complaints,
        "digital_engagement_score": engagement, "sales_channel": channel,
        "last_contact_date": last_contact.strftime("%Y-%m-%d"),
        "churn": np.where(churn, "Yes", "No"),
    })
    # realistic messiness: some missing values
    for col, frac in [("annual_income", .03), ("digital_engagement_score", .05), ("marital_status", .02)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    return df


if __name__ == "__main__":
    train = make(3000, seed=7, id_start=1)
    train.to_csv(OUT / "insurance_churn_train.csv", index=False)

    new = make(300, seed=99, id_start=500001)
    new[["customer_id", "churn"]].to_csv(OUT / "insurance_churn_new_customers_truth.csv", index=False)
    new.drop(columns="churn").to_csv(OUT / "insurance_churn_new_customers.csv", index=False)
    print(f"train churn rate: {(train.churn == 'Yes').mean():.1%}  rows: {len(train)}")
    print(f"new customers: {len(new)}")
