# modules/orders.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from modules.db import get_connection
from modules.visualize import save_figure

def load_order_data(conn=None, chunksize=None):
    """
    Load orders table. For large datasets, pass chunksize
    to load in chunks rather than all at once.
    """
    if conn is None:
        conn = get_connection()

    if chunksize:
        chunks = pd.read_sql(
            "SELECT * FROM orders", conn, chunksize=chunksize
        )
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_sql("SELECT * FROM orders", conn)

    # Coerce numeric columns — SQLite may load empty strings as object dtype
    numeric_cols = [
        'unit_price_usd', 'quantity', 'subtotal_usd',
        'discount_pct', 'discount_amount_usd', 'shipping_fee_usd',
        'tax_pct', 'tax_amount_usd', 'total_amount_usd',
        'delivery_days', 'customer_rating',
        'session_duration_minutes', 'pages_viewed_before_purchase',
        'returned', 'is_repeat_customer'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def temporal_performance(df):
    """
    Revenue and order volume by day of week, month and quarter.
    Identifies momentum patterns and promotional opportunity windows.
    Day of week uses median revenue to reduce outlier sensitivity.
    """
    # Day of week — ordered Mon-Sun
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_df = df.groupby('day_of_week').agg(
        total_orders  =('order_id',        'count'),
        total_revenue =('total_amount_usd','sum'),
        avg_revenue   =('total_amount_usd','mean'),
        median_revenue=('total_amount_usd','median')
    ).round(2)
    day_df = day_df.reindex(
        [d for d in day_order if d in day_df.index]
    )

    # Month
    month_df = df.groupby(['year','month']).agg(
        total_orders  =('order_id',        'count'),
        total_revenue =('total_amount_usd','sum'),
        avg_revenue   =('total_amount_usd','mean')
    ).round(2).reset_index()

    # Quarter
    quarter_df = df.groupby(['year','quarter']).agg(
        total_orders  =('order_id',        'count'),
        total_revenue =('total_amount_usd','sum'),
        avg_revenue   =('total_amount_usd','mean')
    ).round(2).reset_index()

    print("=== TEMPORAL PERFORMANCE ===")
    print("\nBy Day of Week:")
    print(day_df.to_string())
    print("\nBy Quarter:")
    print(quarter_df.to_string(index=False))

    return {
        'day_of_week': day_df,
        'month':       month_df,
        'quarter':     quarter_df
    }

def discount_revenue_analysis(df):
    """
    Correlation between discount_pct and subtotal_usd at order level.
    Extends product-level finding to individual transactions.
    Tests whether discounting drives order value or compresses margin.
    Note: order-level analysis includes repeated customer observations.
    """
    # Overall correlation
    r, p = stats.pearsonr(df['discount_pct'], df['subtotal_usd'])
    sig  = "significant" if p < 0.05 else "not significant"

    print("=== DISCOUNT vs REVENUE (order level) ===")
    print(f"  Overall: r={r:+.3f}  p={p:.4f}  ({sig})")

    # By category
    print("\nBy category:")
    print(f"{'Category':<25} {'r':>8} {'p':>8} {'sig':>16}")
    print("-" * 60)

    results = []
    for cat, group in df.groupby('category'):
        if len(group) > 30:
            r_c, p_c = stats.pearsonr(
                group['discount_pct'], group['subtotal_usd']
            )
            sig_c = "significant" if p_c < 0.05 else "not significant"
            print(f"{cat:<25} {r_c:>+8.3f} {p_c:>8.4f} {sig_c:>16}")
            results.append({
                'category': cat, 'r': round(r_c, 3),
                'p': round(p_c, 4), 'significant': p_c < 0.05
            })

    return pd.DataFrame(results)

def shipping_fee_analysis(df):
    """
    Shipping fee as proportion of subtotal.
    Correlation between subtotal and shipping fee —
    tests whether higher value orders carry disproportionate logistics cost.
    """
    df = df.copy()
    df['shipping_pct_of_subtotal'] = (
        df['shipping_fee_usd'] / df['subtotal_usd'] * 100
    ).round(2)

    r, p = stats.pearsonr(df['subtotal_usd'], df['shipping_fee_usd'])
    sig  = "significant" if p < 0.05 else "not significant"

    print("=== SHIPPING FEE ANALYSIS ===")
    print(f"  Avg shipping fee:              ${df['shipping_fee_usd'].mean():.2f}")
    print(f"  Avg shipping as % of subtotal:  {df['shipping_pct_of_subtotal'].mean():.1f}%")
    print(f"  Median shipping as % of subtotal:{df['shipping_pct_of_subtotal'].median():.1f}%")
    print(f"  Correlation subtotal vs shipping fee:")
    print(f"    r={r:+.3f}  p={p:.4f}  ({sig})")

    # By category
    cat_shipping = df.groupby('category').agg(
        avg_shipping_fee      =('shipping_fee_usd',         'mean'),
        avg_shipping_pct      =('shipping_pct_of_subtotal', 'mean'),
        avg_subtotal          =('subtotal_usd',              'mean')
    ).round(2).sort_values('avg_shipping_pct', ascending=False)

    print("\nShipping burden by category:")
    print(cat_shipping.to_string())

    return df, cat_shipping

def delivery_performance(df):
    """
    Average and median delivery days overall and by category.
    Median used as primary metric — delivery distributions are
    right-skewed by outliers, making mean unreliable.
    """
    overall_mean   = df['delivery_days'].mean()
    overall_median = df['delivery_days'].median()
    overall_std    = df['delivery_days'].std()

    print("=== DELIVERY PERFORMANCE ===")
    print(f"  Overall mean:   {overall_mean:.1f} days")
    print(f"  Overall median: {overall_median:.1f} days")
    print(f"  Std deviation:  {overall_std:.1f} days")

    cat_delivery = df.groupby('category').agg(
        avg_delivery   =('delivery_days','mean'),
        median_delivery=('delivery_days','median'),
        std_delivery   =('delivery_days','std'),
        total_orders   =('order_id',     'count')
    ).round(2).sort_values('median_delivery', ascending=False)

    print("\nDelivery days by category:")
    print(cat_delivery.to_string())

    # Correlation between delivery days and customer rating
    #r, p = stats.pearsonr(df['delivery_days'], df['customer_rating'])
    delivery_clean = df[['delivery_days', 'customer_rating']].apply(
        pd.to_numeric, errors='coerce'
    ).dropna()

    r, p = stats.pearsonr(
        delivery_clean['delivery_days'],
        delivery_clean['customer_rating']
    )
    dropped = len(df) - len(delivery_clean)
    if dropped > 0:
        print(f"  ({dropped} rows dropped due to non-numeric values)")
    sig  = "significant" if p < 0.05 else "not significant"
    print(f"\nDelivery days vs customer rating:")
    print(f"  r={r:+.3f}  p={p:.4f}  ({sig})")

    return cat_delivery

def valued_customer_behavior(df):
    """
    Behavioral profile of high-value customers.
    Aggregates to customer level first to avoid within-customer
    variance inflating correlations at order level.
    Tests: session duration, pages viewed, rating vs order value.
    """
    # Aggregate to customer level
    customer_agg = df.groupby('customer_id').agg(
        avg_order_value       =('total_amount_usd',            'mean'),
        avg_session_duration  =('session_duration_minutes',    'mean'),
        avg_pages_viewed      =('pages_viewed_before_purchase','mean'),
        avg_customer_rating   =('customer_rating',             'mean'),
        total_orders          =('order_id',                    'count')
    ).round(3)

    print("=== VALUED CUSTOMER BEHAVIOR ===")
    print(f"  Customers aggregated: {len(customer_agg):,}")

    behavioral_cols = [
        'avg_session_duration', 'avg_pages_viewed', 'avg_customer_rating'
    ]

    print(f"\n{'Behavioral metric':<25} {'r vs order value':>17} {'p':>8} {'sig':>16}")
    print("-" * 70)

    results = []
    for col in behavioral_cols:
        r, p = stats.pearsonr(
            customer_agg[col], customer_agg['avg_order_value']
        )
        sig = "significant" if p < 0.05 else "not significant"
        print(f"{col:<25} {r:>+17.3f} {p:>8.4f} {sig:>16}")
        results.append({
            'metric': col, 'r': round(r, 3),
            'p': round(p, 4), 'significant': p < 0.05
        })

    # Quartile breakdown — top vs bottom 25% by order value
    customer_agg['value_quartile'] = pd.qcut(
        customer_agg['avg_order_value'], q=4,
        labels=['Q1_low','Q2','Q3','Q4_high']
    )
    quartile_profile = customer_agg.groupby(
        'value_quartile', observed=True
    )[behavioral_cols].mean().round(3)

    print("\nBehavioral profile by order value quartile:")
    print(quartile_profile.to_string())

    return pd.DataFrame(results), quartile_profile

def new_vs_returning(df):
    """
    Revenue and order value comparison between new and returning customers.
    Tests assumption that returning customers spend more per order.
    """
    df = df.copy()
    df['customer_type'] = df['is_repeat_customer'].map(
        {0: 'New', 1: 'Returning'}
    )

    summary = df.groupby('customer_type').agg(
        order_count    =('order_id',        'count'),
        total_revenue  =('total_amount_usd','sum'),
        avg_order_value=('total_amount_usd','mean'),
        median_order   =('total_amount_usd','median'),
        avg_discount   =('discount_pct',    'mean'),
        return_rate    =('returned',        'mean')
    ).round(2)
    summary['revenue_share_pct'] = (
        summary['total_revenue'] / summary['total_revenue'].sum() * 100
    ).round(1)
    summary['return_rate'] = (summary['return_rate'] * 100).round(1)

    # Mann-Whitney U test — are order value distributions different?
    new_vals      = df[df['is_repeat_customer'] == 0]['total_amount_usd']
    returning_vals = df[df['is_repeat_customer'] == 1]['total_amount_usd']
    stat, p = stats.mannwhitneyu(new_vals, returning_vals, alternative='two-sided')
    sig = "significant" if p < 0.05 else "not significant"

    print("=== NEW vs RETURNING CUSTOMERS ===")
    print(summary.to_string())
    print(f"\nMann-Whitney U test (order value distributions):")
    print(f"  U={stat:.0f}  p={p:.4f}  ({sig})")

    return summary

def payment_device_analysis(df):
    """
    Distribution and performance of payment methods and devices.
    Tests whether method or device correlates with order value or returns.
    """
    print("=== PAYMENT METHOD ANALYSIS ===")
    payment = df.groupby('payment_method').agg(
        order_count    =('order_id',        'count'),
        avg_order_value=('total_amount_usd','mean'),
        avg_discount   =('discount_pct',    'mean'),
        return_rate    =('returned',        'mean')
    ).round(2)
    payment['order_share_pct'] = (
        payment['order_count'] / payment['order_count'].sum() * 100
    ).round(1)
    payment['return_rate'] = (payment['return_rate'] * 100).round(1)
    payment = payment.sort_values('avg_order_value', ascending=False)
    print(payment.to_string())

    # Kruskal-Wallis — payment method vs order value
    groups_pay = [
        g['total_amount_usd'].values
        for _, g in df.groupby('payment_method')
    ]
    stat_pay, p_pay = stats.kruskal(*groups_pay)
    sig_pay = "significant" if p_pay < 0.05 else "not significant"
    print(f"\nKruskal-Wallis (order value by payment method):")
    print(f"  H={stat_pay:.3f}  p={p_pay:.4f}  ({sig_pay})")

    print("\n=== DEVICE ANALYSIS ===")
    device = df.groupby('device_used').agg(
        order_count    =('order_id',        'count'),
        avg_order_value=('total_amount_usd','mean'),
        avg_session    =('session_duration_minutes','mean'),
        avg_pages      =('pages_viewed_before_purchase','mean'),
        return_rate    =('returned',        'mean')
    ).round(2)
    device['order_share_pct'] = (
        device['order_count'] / device['order_count'].sum() * 100
    ).round(1)
    device['return_rate'] = (device['return_rate'] * 100).round(1)
    device = device.sort_values('avg_order_value', ascending=False)
    print(device.to_string())

    # Kruskal-Wallis — device vs order value
    groups_dev = [
        g['total_amount_usd'].values
        for _, g in df.groupby('device_used')
    ]
    stat_dev, p_dev = stats.kruskal(*groups_dev)
    sig_dev = "significant" if p_dev < 0.05 else "not significant"
    print(f"\nKruskal-Wallis (order value by device):")
    print(f"  H={stat_dev:.3f}  p={p_dev:.4f}  ({sig_dev})")

    return payment, device

def return_rate_analysis(df):
    """
    Return rate cross-tabulated against category, device,
    payment method and day of week.
    Chi-square tests whether return rate differences are significant.
    """
    print("=== RETURN RATE ANALYSIS ===")

    dimensions = [
        ('category',       'Category'),
        ('device_used',    'Device'),
        ('payment_method', 'Payment Method'),
        ('day_of_week',    'Day of Week')
    ]

    results = {}
    for col, label in dimensions:
        rate = df.groupby(col)['returned'].agg(
            ['mean', 'sum', 'count']
        ).round(4)
        rate.columns    = ['return_rate', 'returns', 'total_orders']
        rate['return_rate'] = (rate['return_rate'] * 100).round(2)
        rate = rate.sort_values('return_rate', ascending=False)

        # Chi-square test
        contingency = pd.crosstab(df[col], df['returned'])
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        sig = "significant" if p < 0.05 else "not significant"

        print(f"\nReturn rate by {label}:")
        print(rate.to_string())
        print(f"  Chi-square: chi2={chi2:.3f}  p={p:.4f}  ({sig})")

        results[col] = rate

    return results

def plot_temporal(temporal_results, save=True):
    """Orders and revenue by day of week."""
    day_df = temporal_results['day_of_week']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(day_df.index, day_df['total_orders'],
                color='steelblue', alpha=0.8)
    axes[0].set_title('Total Orders by Day of Week')
    axes[0].set_xlabel('Day')
    axes[0].set_ylabel('Orders')
    axes[0].tick_params(axis='x', rotation=30)

    axes[1].bar(day_df.index, day_df['total_revenue'],
                color='coral', alpha=0.8)
    axes[1].set_title('Total Revenue by Day of Week')
    axes[1].set_xlabel('Day')
    axes[1].set_ylabel('Revenue (USD)')
    axes[1].tick_params(axis='x', rotation=30)

    plt.suptitle('Temporal Performance — Day of Week', fontsize=12)
    plt.tight_layout()
    if save:
        save_figure('temporal_day_of_week')

def plot_new_vs_returning(summary, save=True):
    """Side-by-side bar chart: new vs returning customer metrics."""
    metrics = ['avg_order_value', 'median_order', 'avg_discount']
    labels  = ['Avg Order Value', 'Median Order', 'Avg Discount %']
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for ax, metric, label in zip(axes, metrics, labels):
        ax.bar(summary.index, summary[metric],
               color=['steelblue', 'coral'], alpha=0.8, width=0.4)
        ax.set_title(label)
        ax.set_ylabel(label)

    plt.suptitle('New vs Returning Customer Profile', fontsize=12)
    plt.tight_layout()
    if save:
        save_figure('new_vs_returning')

def plot_delivery_by_category(cat_delivery, save=True):
    """Horizontal bar chart of median delivery days by category."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cat_delivery_sorted = cat_delivery.sort_values('median_delivery')
    ax.barh(cat_delivery_sorted.index,
            cat_delivery_sorted['median_delivery'],
            color='steelblue', alpha=0.8)
    ax.set_xlabel('Median Delivery Days')
    ax.set_title('Median Delivery Days by Category')
    plt.tight_layout()
    if save:
        save_figure('delivery_by_category')

def plot_return_rates(return_results, save=True):
    """
    Two-panel chart: return rate by category and by device.
    Most actionable dimensions for operational review.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, title in [
        (axes[0], 'category',    'Return Rate by Category'),
        (axes[1], 'device_used', 'Return Rate by Device')
    ]:
        data = return_results[key].sort_values('return_rate', ascending=True)
        ax.barh(data.index, data['return_rate'],
                color='coral', alpha=0.8)
        ax.set_xlabel('Return Rate (%)')
        ax.set_title(title)

    plt.suptitle('Return Rate Analysis', fontsize=12)
    plt.tight_layout()
    if save:
        save_figure('return_rates')

def plot_payment_device(payment, device, save=True):
    """Side-by-side: avg order value by payment method and device."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    pay_sorted = payment.sort_values('avg_order_value', ascending=True)
    axes[0].barh(pay_sorted.index, pay_sorted['avg_order_value'],
                 color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Avg Order Value (USD)')
    axes[0].set_title('Avg Order Value by Payment Method')

    dev_sorted = device.sort_values('avg_order_value', ascending=True)
    axes[1].barh(dev_sorted.index, dev_sorted['avg_order_value'],
                 color='coral', alpha=0.8)
    axes[1].set_xlabel('Avg Order Value (USD)')
    axes[1].set_title('Avg Order Value by Device')

    plt.suptitle('Payment Method and Device Profile', fontsize=12)
    plt.tight_layout()
    if save:
        save_figure('payment_device')

def run_all_visualizations(results):
    """
    Generate and save all order analysis charts in one pass.
    No blocking — all figures saved directly to outputs/figures.
    """
    print("\n=== GENERATING VISUALIZATIONS ===")
    plot_temporal(results['temporal'])
    plot_new_vs_returning(results['new_returning'])
    plot_delivery_by_category(results['delivery'])
    plot_return_rates(results['returns'])
    plot_payment_device(results['payment'], results['device'])
    print("All figures saved.")

def run_order_analysis():
    """Run full orders analysis pipeline."""
    df = load_order_data()
    print(f"Orders loaded: {len(df):,} rows\n")

    temporal = temporal_performance(df)
    print()
    discount = discount_revenue_analysis(df)
    print()
    df_ship, cat_ship = shipping_fee_analysis(df)
    print()
    delivery = delivery_performance(df)
    print()
    behavior, quartiles = valued_customer_behavior(df)
    print()
    new_ret = new_vs_returning(df)
    print()
    pay, dev = payment_device_analysis(df)
    print()
    returns = return_rate_analysis(df)

    results = {
        'data': df,
        'temporal': temporal,
        'discount': discount,
        'shipping': cat_ship,
        'delivery': delivery,
        'behavior': behavior,
        'quartiles': quartiles,
        'new_returning': new_ret,
        'payment': pay,
        'device': dev,
        'returns': returns
    }

    run_all_visualizations(results)
    return results

if __name__ == "__main__":
    results = run_order_analysis()