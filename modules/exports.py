# modules/exports.py
import pandas as pd
from modules.config import DATA_RAW, DATA_EXPORTS
from modules.db import load_table
from datetime import datetime

TABLES = ["product_summary", "customers", "orders", "monthly_revenue"]

def export_tables_to_xlsx():
    """Export all database tables to XLSX for Tableau."""
    for table in TABLES:
        df = load_table(table)
        output_path = DATA_EXPORTS / f"{table}.xlsx"
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"Exported: {output_path} ({len(df)} rows)")

def export_tables_to_csv():
    """Export all database tables to CSV as backup."""
    for table in TABLES:
        df = load_table(table)
        output_path = DATA_EXPORTS / f"{table}.csv"
        df.to_csv(output_path, index=False)
        print(f"Exported: {output_path} ({len(df)} rows)")

def export_ltv_snapshot(df, aggs):
    """
    Export LTV calculation results as timestamped snapshots.
    Preserves historical states for trend analysis.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Customer-level detail
    customer_cols = [
        'customer_id', 'country', 'age_group', 'gender',
        'membership_tier', 'lifespan_years', 'ltv',
        'annual_spend_rate', 'churn_probability',
        'survival_probability', 'projected_ltv'
    ]
    detail_path = DATA_EXPORTS / f"ltv_customers_{timestamp}.csv"
    df[customer_cols].to_csv(detail_path, index=False)
    print(f"Saved: {detail_path}")

    # Aggregated snapshots per dimension
    for dimension, agg_df in aggs.items():
        agg_path = DATA_EXPORTS / f"ltv_by_{dimension}_{timestamp}.csv"
        agg_df.to_csv(agg_path)
        print(f"Saved: {agg_path}")

def store_ltv_snapshot(df, conn=None):
    """
    Store LTV results as a versioned snapshot in the database.
    Each run appends a new snapshot rather than overwriting.
    """
    from modules.db import get_connection

    if conn is None:
        conn = get_connection()

    snapshot = df[[
        'customer_id', 'lifespan_years', 'ltv',
        'annual_spend_rate', 'churn_probability',
        'survival_probability', 'projected_ltv'
    ]].copy()

    snapshot['snapshot_date'] = datetime.now().strftime("%Y-%m-%d")
    snapshot['pipeline_version'] = '1.0'

    snapshot.to_sql(
        'ltv_snapshots',
        conn,
        if_exists='append',   # append preserves history
        index=False
    )
    print(f"Stored {len(snapshot)} rows to ltv_snapshots table.")

if __name__ == "__main__":
    export_tables_to_xlsx()