# main.py
"""
E-Commerce Analysis Pipeline
=============================
Orchestrates the full analytical workflow across four datasets:
product_summary, customers, orders, monthly_revenue.

Usage:
    python main.py                  # run full pipeline
    python main.py --module products  # run single module
    python main.py --no-viz         # skip visualizations

Modules:
    validate   — data quality checks across all tables
    products   — product performance, ABC classification, discount analysis
    customers  — segmentation, churn, LTV, acquisition channel
    orders     — temporal patterns, behavioral analysis, return rates
    exports    — XLSX export for Tableau, LTV snapshots
"""

import argparse
import time
from datetime import datetime

# ── module imports ────────────────────────────────────────────────────────────
from modules.validate  import run_all_checks
from modules.products  import run_product_analysis
from modules.customers import run_customer_analysis, run_extended_customer_analysis
from modules.ltv       import run_ltv_pipeline
from modules.orders    import run_order_analysis
from modules.exports   import export_tables_to_xlsx, export_ltv_snapshot
from modules.db        import get_connection

# ── helpers ───────────────────────────────────────────────────────────────────
def section(title):
    """Print a formatted section header."""
    width = 60
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")

def elapsed(start):
    """Return elapsed time as a readable string."""
    seconds = time.time() - start
    return f"{seconds:.1f}s"

# ── pipeline stages ───────────────────────────────────────────────────────────
def stage_validate():
    section("STAGE 1 — DATA VALIDATION")
    start = time.time()
    quality_df = run_all_checks()
    print(f"\nValidation completed in {elapsed(start)}")
    return quality_df

def stage_products():
    section("STAGE 2 — PRODUCT ANALYSIS")
    start = time.time()
    results = run_product_analysis()
    print(f"\nProduct analysis completed in {elapsed(start)}")
    return results

def stage_customers():
    section("STAGE 3 — CUSTOMER ANALYSIS")
    start = time.time()
    conn = get_connection()

    # Core segmentation and churn
    core = run_customer_analysis()

    # Extended: satisfaction, activity, channel, churn model
    extended = run_extended_customer_analysis(core['data'])

    # LTV pipeline
    section("STAGE 3b — LTV PIPELINE")
    ltv_df, ltv_aggs = run_ltv_pipeline()

    print(f"\nCustomer + LTV analysis completed in {elapsed(start)}")
    return {
        'core':     core,
        'extended': extended,
        'ltv_df':   ltv_df,
        'ltv_aggs': ltv_aggs
    }

def stage_orders():
    section("STAGE 4 — ORDER ANALYSIS")
    start = time.time()
    results = run_order_analysis()
    print(f"\nOrder analysis completed in {elapsed(start)}")
    return results

def stage_exports(ltv_df, ltv_aggs):
    section("STAGE 5 — EXPORTS")
    start = time.time()

    # XLSX for Tableau
    export_tables_to_xlsx()

    # LTV snapshot — CSV and SQLite
    export_ltv_snapshot(ltv_df, ltv_aggs)

    print(f"\nExports completed in {elapsed(start)}")

# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='E-Commerce Analysis Pipeline'
    )
    parser.add_argument(
        '--module',
        choices=['validate', 'products', 'customers', 'orders', 'exports'],
        help='Run a single module instead of the full pipeline'
    )
    parser.add_argument(
        '--skip-validate',
        action='store_true',
        help='Skip validation stage (faster iteration during development)'
    )
    parser.add_argument(
        '--skip-exports',
        action='store_true',
        help='Skip export stage'
    )
    return parser.parse_args()

# ── entry point ───────────────────────────────────────────────────────────────
def main():
    args       = parse_args()
    start_time = time.time()

    print(f"\nE-Commerce Analysis Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Single module mode
    if args.module:
        section(f"RUNNING MODULE: {args.module.upper()}")
        if args.module == 'validate':
            stage_validate()
        elif args.module == 'products':
            stage_products()
        elif args.module == 'customers':
            stage_customers()
        elif args.module == 'orders':
            stage_orders()
        elif args.module == 'exports':
            # Exports need LTV data — run LTV pipeline first
            ltv_df, ltv_aggs = run_ltv_pipeline()
            stage_exports(ltv_df, ltv_aggs)
        print(f"\nCompleted in {elapsed(start_time)}")
        return

    # Full pipeline
    if not args.skip_validate:
        stage_validate()

    product_results  = stage_products()
    customer_results = stage_customers()
    order_results    = stage_orders()

    if not args.skip_exports:
        stage_exports(
            customer_results['ltv_df'],
            customer_results['ltv_aggs']
        )

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {elapsed(start_time)}")
    print(f"  Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()