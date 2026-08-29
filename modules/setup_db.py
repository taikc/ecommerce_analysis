# setup_db.py
"""
Run this once after cloning to build the SQLite database from source CSVs.
Place CSV files in data/raw/ before running.

Usage:
    python setup_db.py
"""

import sqlite3
import pandas as pd
from pathlib import Path
from modules.config import DATA_RAW, DB_PATH

def setup():
    tables = {
        "product_summary":  DATA_RAW / "product_summary.csv",
        "customers":        DATA_RAW / "customers.csv",
        "orders":           DATA_RAW / "orders.csv",
        "monthly_revenue":  DATA_RAW / "monthly_revenue.csv"
    }

    missing = [name for name, path in tables.items() if not path.exists()]
    if missing:
        print(f"Missing CSV files in data/raw/: {missing}")
        print("Download from Kaggle and place in data/raw/ before running.")
        return

    conn = sqlite3.connect(DB_PATH)
    for table_name, csv_path in tables.items():
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Loaded {table_name}: {len(df):,} rows")

    conn.close()
    print(f"\nDatabase created at: {DB_PATH}")

if __name__ == "__main__":
    setup()