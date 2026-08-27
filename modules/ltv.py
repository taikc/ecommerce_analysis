# modules/ltv.py
import pandas as pd
import numpy as np
from datetime import datetime
from modules.db import get_connection

# Customer Lifetime Value (CLV)
# Extract registration_date and churned from customers, and transaction detail from orders for cross-validation
def load_customer_data(conn=None):
    if conn is None:
        conn = get_connection()
    return pd.read_sql("""
        SELECT
            customer_id, country, age, gender,
            membership_tier, registration_date,
            total_orders, total_spend_usd,
            avg_order_value_usd, days_since_last_purchase,
            churned
        FROM customers
    """, conn)

def calculate_lifespan(df):
    """Calculate realized lifespan per customer.
    Active customers: registration to today
    Churned customers: registration to last known purchase"""
    today = pd.Timestamp(datetime.today().date())
    df = df.copy()
    df['registration_date'] = pd.to_datetime(df['registration_date'])
    df['last_purchase_date'] = (
        today - pd.to_timedelta(df['days_since_last_purchase'], unit='D')
    )
    df['lifespan_days'] = df.apply(
        lambda row: (row['last_purchase_date'] - row['registration_date']).days
        if row['churned'] == 1
        else (today - row['registration_date']).days,
        axis=1
    )
    df['lifespan_years'] = (df['lifespan_days'].clip(lower=1) / 365).round(4)
    return df

def calculate_churn_probability(df):
    """
    Estimate churn probability for active customers
    based on recency distribution.
    Uses statistical thresholds rather than arbitrary cutoffs.
    """
    import numpy as np

    # Use only active customers to build the recency distribution
    active = df[df['churned'] == 0]['days_since_last_purchase']

    # Statistical thresholds from the distribution itself
    p50 = active.quantile(0.50)  # median — baseline activity
    p75 = active.quantile(0.75)  # elevated risk
    p90 = active.quantile(0.90)  # high risk

    print(f"Recency thresholds (active customers):")
    print(f"  Median (p50): {p50:.0f} days")
    print(f"  p75:          {p75:.0f} days")
    print(f"  p90:          {p90:.0f} days")

    # Assign churn probability based on recency bucket
    def churn_prob(row):
        if row['churned'] == 1:
            return 1.0  # already churned — certain
        d = row['days_since_last_purchase']
        if d <= p50:
            return 0.05  # recently active — low risk
        elif d <= p75:
            return 0.25  # moderate recency — elevated risk
        elif d <= p90:
            return 0.60  # high recency — high risk
        else:
            return 0.90  # extreme recency — near certain churn

    df = df.copy()
    df['churn_probability'] = df.apply(churn_prob, axis=1)

    return df

def calculate_projected_ltv(df, projection_years=3):
    """
    Project LTV forward, discounted by churn probability.
    Churned customers return realized LTV only.
    Active customers are discounted by survival probability.
    """
    df = df.copy()

    # Realized LTV — total spend to date
    df['ltv'] = df['total_spend_usd'].round(2)

    # Survival probability = likelihood of remaining active
    df['survival_probability'] = 1 - df['churn_probability']

    # Annual spend rate from realized history
    df['annual_spend_rate'] = (
            df['total_spend_usd'] / df['lifespan_years']
    ).round(2)

    # Projected LTV — discounted by survival probability
    df['projected_ltv'] = df.apply(
        lambda row: (
                row['total_spend_usd'] +
                row['annual_spend_rate'] * projection_years * row['survival_probability']
        ) if row['churned'] == 0
        else row['total_spend_usd'],
        axis=1
    ).round(2)
    
    return df

def add_age_group(df):
    """Bin age into labeled groups."""
    df = df.copy()
    df['age_group'] = pd.cut(
        df['age'],
        bins=[17, 24, 31, 38, 45, 52, 59, 100],
        labels=['18-24','25-31','32-38','39-45','46-52','53-59','60+']
    )
    return df

def aggregate_ltv(df):
    """Aggregate LTV metrics by tier, country and age group."""
    def agg(groupby_col):
        return df.groupby(groupby_col, observed=True).agg(
            customer_count  =('customer_id',      'count'),
            avg_ltv         =('ltv',               'mean'),
            median_ltv      =('ltv',               'median'),
            avg_projected_ltv=('projected_ltv',    'mean'),
            avg_annual_rate =('annual_spend_rate', 'mean'),
            avg_churn_prob  =('churn_probability', 'mean')
        ).round(2)

    tier    = agg('membership_tier').reindex(
        ['Free','Silver','Gold','Platinum']
    )
    country = agg('country').sort_values('avg_projected_ltv', ascending=False)
    age     = agg('age_group')

    return {'tier': tier, 'country': country, 'age_group': age}

def run_ltv_pipeline(projection_years=3):
    """Run full LTV pipeline and return enriched customer dataframe."""
    df = load_customer_data()
    df = calculate_lifespan(df)
    df = calculate_churn_probability(df)
    df = calculate_projected_ltv(df, projection_years)
    df = add_age_group(df)

    aggregations = aggregate_ltv(df)

    print("\n=== LTV BY MEMBERSHIP TIER ===")
    print(aggregations['tier'].to_string())
    print("\n=== LTV BY COUNTRY (top 10) ===")
    print(aggregations['country'].head(10).to_string())
    print("\n=== LTV BY AGE GROUP ===")
    print(aggregations['age_group'].to_string())

    return df, aggregations

if __name__ == "__main__":
    df, aggs = run_ltv_pipeline()