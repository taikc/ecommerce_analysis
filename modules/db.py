import sqlite3
import pandas as pd
from modules.config import DB_PATH

def get_connection():
    return sqlite3.connect(DB_PATH)

def query(sql):
    with get_connection() as conn:
        return pd.read_sql(sql, conn)

def load_table(table_name):
    return query(f"SELECT * FROM {table_name}")
