import pandas as pd
from modules.db import get_connection, load_table

TABLES = ["product_summary", "customers", "orders", "monthly_revenue"]

def check_nulls_and_duplicates():
    """Check all tables for nulls and duplicates."""
    quality_rows = []
    
    for table in TABLES:
        df = load_table(table)
        for col in df.columns:
            quality_rows.append({
                "table":      table,
                "column":     col,
                "dtype":      str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_pct":   round(df[col].isnull().sum() / len(df) * 100, 2),
                "duplicates": int(df.duplicated().sum()),
                "row_count":  len(df)
            })
    
    return pd.DataFrame(quality_rows)

def check_referential_integrity():
    """Check foreign key relationships across tables."""
    results = {}
    
    with get_connection() as conn:
        # Orders → Customers
        orphan_customers = pd.read_sql("""
            SELECT COUNT(DISTINCT o.customer_id) AS orphaned
            FROM orders o
            LEFT JOIN customers c ON o.customer_id = c.customer_id
            WHERE c.customer_id IS NULL
        """, conn).iloc[0, 0]
        
        # Orders → Products
        orphan_products = pd.read_sql("""
            SELECT COUNT(DISTINCT o.product_name) AS orphaned
            FROM orders o
            LEFT JOIN product_summary p ON o.product_name = p.product_name
            WHERE p.product_name IS NULL
        """, conn).iloc[0, 0]
        
        # Category consistency
        orphan_categories = pd.read_sql("""
            SELECT COUNT(DISTINCT o.category) AS orphaned
            FROM orders o
            LEFT JOIN product_summary p ON o.category = p.category
            WHERE p.category IS NULL
        """, conn).iloc[0, 0]
    
    results['orphaned_customer_ids'] = orphan_customers
    results['orphaned_product_names'] = orphan_products
    results['orphaned_categories']    = orphan_categories
    
    return results

def run_all_checks():
    """Run full validation suite and print summary."""
    print("=== NULL AND DUPLICATE CHECK ===")
    quality_df = check_nulls_and_duplicates()
    issues = quality_df[
        (quality_df['null_count'] > 0) |
        (quality_df['duplicates'] > 0)
    ]
    if issues.empty:
        print("All tables clean — no nulls or duplicates found.")
    else:
        print(issues.to_string(index=False))
    
    print("\n=== REFERENTIAL INTEGRITY CHECK ===")
    integrity = check_referential_integrity()
    for check, count in integrity.items():
        status = "PASS" if count == 0 else f"FAIL ({count} orphaned)"
        print(f"  {check:<30} {status}")
    
    return quality_df

if __name__ == "__main__":
    run_all_checks()
