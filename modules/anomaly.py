# modules/anomaly.py
"""
Anomaly Detection Addendum
===========================
Flags suspicious return and order patterns using three complementary
approaches: rule-based thresholds, statistical process control,
and Isolation Forest.

Epistemic note: this module operates on synthetic data generated from
uniform distributions. Flagged cases demonstrate methodology, not actual
fraud signals. Thresholds are calibrated from the data's own distribution
rather than domain benchmarks — in production, thresholds would be
validated against confirmed fraud cases and adjusted for base rates.

False positive / false negative tradeoff is explicit throughout:
- Lower thresholds catch more true anomalies but generate more
  investigation workload (false positives)
- Higher thresholds reduce workload but miss real signals
  (false negatives)
- The appropriate balance depends on the cost of each error type,
  which is a business decision, not a statistical one.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from sklearn.ensemble     import IsolationForest
from sklearn.preprocessing import StandardScaler
from modules.db           import get_connection
from modules.visualize    import save_figure


# ── data preparation ──────────────────────────────────────────────────────────

def load_anomaly_data(conn=None, min_orders=3):
    """
    Build customer-level behavioral profile for anomaly detection.

    min_orders parameter enforces minimum purchase history before
    return rate flags are meaningful. Customers below this threshold
    are retained in the dataset but excluded from return-rate based
    flags — their behavioral history is insufficient to distinguish
    pattern from noise.

    Behavioral signatures targeted:
    - Wardrobing: high return rate on high-value orders (min 3 orders)
    - Refund velocity: return frequency relative to account window
    - Payment cycling: multiple payment methods on one account
    - Order velocity: rapid ordering relative to account age
    """
    if conn is None:
        conn = get_connection()

    df = pd.read_sql("""
        SELECT
            o.customer_id,
            COUNT(o.order_id)                              AS total_orders,
            SUM(CASE WHEN o.returned = 1 THEN 1 ELSE 0 END)
                                                           AS total_returns,
            ROUND(AVG(o.total_amount_usd), 2)              AS avg_order_value,
            ROUND(SUM(o.total_amount_usd), 2)              AS total_spend,
            ROUND(AVG(
                CASE WHEN o.returned = 1
                THEN o.total_amount_usd END), 2)           AS avg_returned_value,
            ROUND(AVG(o.discount_pct), 2)                  AS avg_discount_pct,
            COUNT(DISTINCT o.payment_method)               AS payment_methods_used,
            COUNT(DISTINCT o.device_used)                  AS devices_used,
            MIN(o.order_date)                              AS first_order_date,
            MAX(o.order_date)                              AS last_order_date,
            c.membership_tier,
            c.country,
            c.churned
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY o.customer_id
    """, conn)

    # Date features
    df['first_order_date'] = pd.to_datetime(df['first_order_date'])
    df['last_order_date']  = pd.to_datetime(df['last_order_date'])
    df['account_age_days'] = (
        df['last_order_date'] - df['first_order_date']
    ).dt.days.clip(lower=1)

    # Return rate — only meaningful above min_orders threshold
    df['return_rate'] = np.where(
        df['total_orders'] >= min_orders,
        df['total_returns'] / df['total_orders'],
        np.nan          # NaN signals insufficient history
    )

    # Refund velocity — returns per 30-day window
    # Captures acceleration of returns regardless of total count
    df['refund_velocity'] = np.where(
        df['total_returns'] >= 2,  # minimum 2 returns before velocity matters
        df['total_returns'] / (df['account_age_days'] / 30),
        0.0  # zero velocity for single-return customers
    )

    # Wardrobing index — return rate weighted by avg returned value
    # High rate on high-value items is more suspicious than
    # high rate on low-value items
    df['wardrobing_index'] = np.where(
        df['total_orders'] >= min_orders,
        df['return_rate'] * df['avg_returned_value'].fillna(0),
        np.nan
    )

    # Order velocity — orders per 30-day window
    df['order_velocity'] = (
        df['total_orders'] / (df['account_age_days'] / 30)
    ).round(4)

    # Payment cycling flag — raw feature for rule engine
    # 3+ payment methods on one account is unusual
    # Depends on total_orders count per customer_id
    df['payment_velocity'] = (
            df['payment_methods_used'] / np.log1p(df['total_orders'])
    ).round(4)

    # History depth label — used in reporting to contextualize flags
    df['history_depth'] = pd.cut(
        df['total_orders'],
        bins=[0, 2, 5, 15, np.inf],
        labels=['thin (<3)', 'low (3-5)', 'medium (6-15)', 'deep (16+)']
    )

    print(f"Customers loaded: {len(df):,}")
    print(f"History depth distribution:")
    print(df['history_depth'].value_counts().sort_index().to_string())
    print(f"\nCustomers with sufficient history for return rate flags "
          f"(≥{min_orders} orders): "
          f"{df['total_orders'].ge(min_orders).sum():,} "
          f"({df['total_orders'].ge(min_orders).mean()*100:.1f}%)")

    return df


# ── baseline statistics ───────────────────────────────────────────────────────

def compute_baselines(df):
    """
    Establish baselines for all flaggable metrics.
    return_rate and wardrobing_index computed on eligible
    customers only (sufficient history) to avoid base rate
    contamination from thin-history accounts.
    """
    # Use only eligible customers for return-rate based metrics
    eligible = df[df['return_rate'].notna()]

    metric_sources = {
        'return_rate':      eligible['return_rate'],
        'wardrobing_index': eligible['wardrobing_index'],
        'refund_velocity':  df['refund_velocity'],
        'order_velocity':   df['order_velocity'],
        'avg_order_value':  df['avg_order_value'],
        'payment_methods_used': df['payment_methods_used']
    }

    baselines = {}
    print("=== BASELINE STATISTICS ===")
    print(f"\n{'Metric':<25} {'N':>6} {'Mean':>8} {'Median':>8} "
          f"{'Std':>8} {'P90':>8} {'P95':>8}")
    print("-" * 80)

    for metric, series in metric_sources.items():
        series = series.dropna()
        b = {
            'mean':   series.mean(),
            'median': series.median(),
            'std':    series.std(),
            'p75':    series.quantile(0.75),
            'p90':    series.quantile(0.90),
            'p95':    series.quantile(0.95),
            'p99':    series.quantile(0.99),
            'iqr':    series.quantile(0.75) - series.quantile(0.25)
        }
        baselines[metric] = b
        print(f"{metric:<25} {len(series):>6} {b['mean']:>8.3f} "
              f"{b['median']:>8.3f} {b['std']:>8.3f} "
              f"{b['p90']:>8.3f} {b['p95']:>8.3f}")

    eligible_n = df['return_rate'].notna().sum()
    print(f"\nBase return rate (eligible customers, n={eligible_n}): "
          f"{eligible['return_rate'].mean():.3f} "
          f"({eligible['return_rate'].mean()*100:.1f}%)")
    print(f"Thin-history customers excluded from return rate flags: "
          f"{df['return_rate'].isna().sum():,}")

    return baselines


# ── approach 1: rule-based flags ──────────────────────────────────────────────

def rule_based_flags(df, baselines, strictness='moderate'):
    """
    Flag customers by behavioral signature rather than raw thresholds.
    Return rate flags only applied to customers with sufficient history.
    Four distinct fraud pattern types targeted.
    """
    df = df.copy()

    rb  = baselines.get('return_rate', {})
    rv  = baselines.get('refund_velocity', {})
    ov  = baselines.get('order_velocity', {})
    wi  = baselines.get('wardrobing_index', {})

    if strictness == 'loose':
        rr_threshold  = rb.get('p90', 0.333)
        rv_threshold  = rv.get('p90', 1.0)
        ov_threshold  = ov.get('p95', 30.0)
        wi_threshold  = wi.get('p90', 50.0)
    elif strictness == 'strict':
        rr_threshold  = rb.get('p95', 0.500)
        rv_threshold  = rv.get('p95', 2.0)
        ov_threshold  = ov.get('p99', 30.0)
        wi_threshold  = wi.get('p95', 100.0)
    else:  # moderate
        rr_threshold  = rb.get('p90', 0.333) + 1.5 * rb.get('iqr', 0.1)
        rv_threshold  = rv.get('p90', 1.0)   + 1.5 * rv.get('iqr', 0.5)
        ov_threshold  = ov.get('p90', 20.0)  + 1.5 * ov.get('iqr', 5.0)
        wi_threshold  = wi.get('p90', 50.0)  + 1.5 * wi.get('iqr', 20.0)

    # Flag 1 — Wardrobing: high return rate on sufficient history
    # NaN return_rate = insufficient history = no flag
    df['flag_wardrobing'] = (
        df['return_rate'].notna() &
        (df['return_rate'] > rr_threshold)
    )

    # Flag 2 — Wardrobing index: high-value return pattern
    df['flag_high_value_returns'] = (
        df['wardrobing_index'].notna() &
        (df['wardrobing_index'] > wi_threshold)
    )

    # Flag 3 — Refund velocity: accelerating return frequency
    df['flag_refund_velocity'] = df['refund_velocity'] > rv_threshold

    # Flag 4 — Order velocity spike: rapid ordering
    df['flag_order_velocity'] = df['order_velocity'] > ov_threshold

    # Flag 5 — Payment cycling: multiple payment methods
    pv_threshold = df['payment_velocity'].quantile(0.90)
    df['flag_payment_cycling'] = df['payment_velocity'] > pv_threshold

    df['rule_flag_count'] = (
        df['flag_wardrobing'].astype(int) +
        df['flag_high_value_returns'].astype(int) +
        df['flag_refund_velocity'].astype(int) +
        df['flag_order_velocity'].astype(int) +
        df['flag_payment_cycling'].astype(int)
    )
    df['rule_flagged'] = df['rule_flag_count'] > 0

    print(f"\n=== RULE-BASED FLAGS (strictness: {strictness}) ===")
    print(f"  Thresholds applied:")
    print(f"    Wardrobing (return rate):    > {rr_threshold:.3f}  "
          f"[history ≥ 3 orders only]")
    print(f"    High-value returns (index):  > {wi_threshold:.3f}")
    print(f"    Refund velocity:             > {rv_threshold:.3f} /30d")
    print(f"    Order velocity:              > {ov_threshold:.3f} /30d")
    print(f"    Payment cycling:             ≥ 3 payment methods")

    flag_cols = {
        'flag_wardrobing':          'Wardrobing',
        'flag_high_value_returns':  'High-value returns',
        'flag_refund_velocity':     'Refund velocity',
        'flag_order_velocity':      'Order velocity',
        'flag_payment_cycling':     'Payment cycling'
    }

    print(f"\n  Flagged by type:")
    for col, label in flag_cols.items():
        count = df[col].sum()
        pct   = count / len(df) * 100
        print(f"    {label:<25} {count:>5} ({pct:.1f}%)")

    print(f"\n  Total flagged (any rule): "
          f"{df['rule_flagged'].sum()} "
          f"({df['rule_flagged'].mean()*100:.1f}%)")
    print(f"  Flagged by 2+ rules:      "
          f"{(df['rule_flag_count'] >= 2).sum()} "
          f"(higher confidence)")

    return df


# ── approach 2: statistical process control ───────────────────────────────────

def statistical_process_control(df, baselines, z_threshold=2.5):
    """
    Z-score based control limits. Flags customers whose metrics
    fall outside z_threshold standard deviations from the mean.

    Z-threshold selection:
    - z=2.0: flags ~4.6% of population (95.4% within limits)
    - z=2.5: flags ~1.2% of population (98.8% within limits)
    - z=3.0: flags ~0.3% of population (99.7% within limits)

    Default z=2.5 is a moderate calibration — analogous to a
    seasonality-adjusted tolerance band that accepts more variance
    than a pure 2-sigma rule but tighter than 3-sigma.

    IQR fence included as a non-parametric complement — robust
    to skewed distributions where Z-score assumptions break down.
    """
    df = df.copy()
    spc_cols = ['return_rate', 'avg_order_value',
                'order_velocity', 'avg_discount_pct']

    exclude = {'payment_methods_used'}
    spc_cols = [
        col for col in baselines.keys()
        if col not in exclude
           and col in df.columns
    ]

    for col in spc_cols:
        mean = baselines[col]['mean']
        std  = baselines[col]['std']
        iqr  = baselines[col]['iqr']
        p75  = baselines[col]['p75']

        # Z-score flag — upper tail only (we flag high, not low)
        df[f'z_{col}'] = ((df[col] - mean) / std).round(3)
        df[f'zscore_flag_{col}'] = df[f'z_{col}'] > z_threshold

        # IQR fence flag — Tukey's method
        upper_fence = p75 + 3.0 * iqr
        df[f'iqr_flag_{col}'] = df[col] > upper_fence

    # Composite SPC flag — flagged by either method on any metric
    zscore_cols = [c for c in df.columns if c.startswith('zscore_flag_')]
    iqr_cols    = [c for c in df.columns if c.startswith('iqr_flag_')]

    df['spc_flag_count'] = df[zscore_cols + iqr_cols].sum(axis=1)
    df['spc_flagged']    = df['spc_flag_count'] > 0

    print(f"\n=== STATISTICAL PROCESS CONTROL (z={z_threshold}) ===")
    for col in spc_cols:
        z_count   = df[f'zscore_flag_{col}'].sum()
        iqr_count = df[f'iqr_flag_{col}'].sum()
        print(f"  {col:<25} Z-flag: {z_count:>4}  IQR-flag: {iqr_count:>4}")

    print(f"\n  Total SPC-flagged: {df['spc_flagged'].sum()} "
          f"({df['spc_flagged'].mean()*100:.1f}%)")

    return df


# ── approach 3: isolation forest ─────────────────────────────────────────────

def isolation_forest_flags(df, contamination=0.05, random_state=42):
    """
    Unsupervised multivariate anomaly detection.
    Detects combinations of features that are jointly unusual —
    catches cases where no single metric exceeds a threshold
    but the overall behavioral profile is atypical.

    Contamination parameter = assumed proportion of anomalies
    in the dataset. Set to 0.05 (5%) as a conservative estimate.
    In production, this would be calibrated against confirmed
    fraud base rates — typically 0.1-2% in real transaction data,
    meaning 5% is deliberately permissive to reduce false negatives
    at the cost of more investigation workload.

    Note: IsolationForest is sensitive to feature scale —
    StandardScaler applied before fitting.
    """
    features = [
        'return_rate', 'avg_order_value', 'order_velocity',
        'avg_discount_pct', 'payment_methods_used',
        'return_value_ratio', 'total_orders'
    ]

    features = [f for f in features if f in df.columns and
                df[f].notna().sum() > len(df) * 0.5]
    print(f"  Features used: {features}")

    feature_df = df[features].fillna(0)
    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(feature_df)

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200
    )
    iso.fit(X_scaled)

    # -1 = anomaly, 1 = normal — remap to boolean
    df = df.copy()
    df['iso_prediction']    = iso.predict(X_scaled)
    df['iso_flagged']       = df['iso_prediction'] == -1
    df['iso_anomaly_score'] = iso.decision_function(X_scaled).round(4)
    # More negative score = more anomalous

    print(f"\n=== ISOLATION FOREST (contamination={contamination}) ===")
    print(f"  Flagged: {df['iso_flagged'].sum()} customers "
          f"({df['iso_flagged'].mean()*100:.1f}%)")
    print(f"  Anomaly score range: "
          f"{df['iso_anomaly_score'].min():.3f} to "
          f"{df['iso_anomaly_score'].max():.3f}")
    print(f"  (More negative = more anomalous)")

    return df


# ── consensus analysis ────────────────────────────────────────────────────────

def consensus_analysis(df):
    """
    Cross-method agreement analysis.
    Cases flagged by all three approaches are higher-confidence
    anomalies — the methods are complementary, not redundant.

    Rule-based: catches known pattern types (high return rate)
    SPC:        catches univariate statistical outliers
    IsoForest:  catches multivariate behavioral anomalies

    Agreement across methods reduces false positive risk without
    requiring labeled ground truth — a practical calibration
    strategy when confirmed fraud cases are unavailable.
    """
    df = df.copy()
    df['methods_flagged'] = (
        df['rule_flagged'].astype(int) +
        df['spc_flagged'].astype(int) +
        df['iso_flagged'].astype(int)
    )

    df['consensus_tier'] = pd.cut(
        df['methods_flagged'],
        bins=[-1, 0, 1, 2, 3],
        labels=['Clean', 'Low', 'Medium', 'High']
    )

    print("\n=== CONSENSUS ANALYSIS ===")
    tier_counts = df['consensus_tier'].value_counts().sort_index()
    total       = len(df)

    for tier, count in tier_counts.items():
        pct = count / total * 100
        print(f"  {tier:<8}: {count:>5} customers ({pct:.1f}%)")

    print(f"\n  High confidence (all 3 methods): "
          f"{(df['methods_flagged'] == 3).sum()} customers")

    # Profile of high-confidence flags
    high = df[df['methods_flagged'] == 3]
    if len(high) > 0:
        print(f"\n  High-confidence flag profile:")
        print(f"    Avg return rate:    {high['return_rate'].mean():.3f} "
              f"(vs {df['return_rate'].mean():.3f} population)")
        print(f"    Avg order value:    ${high['avg_order_value'].mean():.2f} "
              f"(vs ${df['avg_order_value'].mean():.2f} population)")
        print(f"    Avg order velocity: {high['order_velocity'].mean():.3f} "
              f"(vs {df['order_velocity'].mean():.3f} population)")
        print(f"    Churn rate:         "
              f"{high['churned'].mean()*100:.1f}% "
              f"(vs {df['churned'].mean()*100:.1f}% population)")

    return df


# ── false positive / negative tradeoff ───────────────────────────────────────

def tradeoff_analysis(df, baselines):
    """
    Explicit false positive / false negative tradeoff across
    threshold levels. Demonstrates that threshold selection is
    a business decision, not a statistical one.

    Frames the tradeoff in operational terms:
    - False positives = unnecessary investigations (analyst time cost)
    - False negatives = missed anomalies (potential loss cost)

    Without confirmed fraud labels, precision and recall cannot be
    calculated directly. Instead, we show how flag volume and
    average severity change across threshold levels — the
    operational inputs to a cost-benefit decision.
    """
    print("\n=== FALSE POSITIVE / NEGATIVE TRADEOFF ===")
    print(f"\n  Context: population base return rate = "
          f"{df['return_rate'].mean()*100:.1f}%")
    print(f"  Without confirmed fraud labels, we cannot calculate")
    print(f"  true precision or recall. The table below shows")
    print(f"  flag volume and average flagged return rate across")
    print(f"  threshold levels — the operational inputs to a")
    print(f"  cost-benefit threshold decision.\n")

    rb = baselines['return_rate']
    thresholds = {
        'Very strict (p99)': rb['p95'] + 2 * rb['std'],
        'Strict (p95)':      rb['p95'],
        'Moderate (p90+IQR)':rb['p90'] + 1.5 * rb['iqr'],
        'Loose (p90)':       rb['p90'],
        'Very loose (p75)':  rb['p75']
    }

    print(f"  {'Threshold level':<25} {'Cutoff':>8} "
          f"{'Flagged':>8} {'Flag%':>7} {'Avg RR flagged':>15}")
    print("  " + "-" * 70)

    for label, cutoff in thresholds.items():
        flagged    = df[df['return_rate'] > cutoff]
        flag_count = len(flagged)
        flag_pct   = flag_count / len(df) * 100
        avg_rr     = flagged['return_rate'].mean() if flag_count > 0 else 0
        print(f"  {label:<25} {cutoff:>8.3f} "
              f"{flag_count:>8} {flag_pct:>6.1f}% "
              f"{avg_rr:>14.3f}")

    print(f"\n  Interpretation guide:")
    print(f"  - Higher thresholds: fewer flags, higher avg severity")
    print(f"    → Lower analyst workload, higher false negative risk")
    print(f"  - Lower thresholds: more flags, lower avg severity")
    print(f"    → Higher analyst workload, lower false negative risk")
    print(f"  - Optimal threshold depends on: cost of investigation vs")
    print(f"    cost of missed anomaly — a business input, not a")
    print(f"    statistical one.")


# ── visualizations ────────────────────────────────────────────────────────────

def plot_anomaly_overview(df, baselines, save=True):
    """
    Four-panel overview:
    1. Return rate distribution with threshold bands
    2. Consensus tier distribution
    3. Anomaly score vs return rate (Isolation Forest)
    4. Flag overlap heatmap across methods
    """
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1 — return rate distribution with thresholds
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['return_rate'], bins=40,
             color='steelblue', alpha=0.7, edgecolor='white')
    rb = baselines['return_rate']
    for label, val, color in [
        ('p75',  rb['p75'],  'gold'),
        ('p90',  rb['p90'],  'orange'),
        ('p95',  rb['p95'],  'red'),
    ]:
        ax1.axvline(val, color=color, linestyle='--',
                    linewidth=1.5, label=label)
    ax1.set_xlabel('Return Rate')
    ax1.set_ylabel('Customer Count')
    ax1.set_title('Return Rate Distribution\nwith Threshold Bands')
    ax1.legend(fontsize=8)

    # Panel 2 — consensus tier counts
    ax2 = fig.add_subplot(gs[0, 1])
    tier_counts = df['consensus_tier'].value_counts().sort_index()
    colors_tier = ['steelblue', 'gold', 'orange', 'red']
    ax2.bar(tier_counts.index, tier_counts.values,
            color=colors_tier[:len(tier_counts)], alpha=0.8)
    ax2.set_xlabel('Consensus Tier')
    ax2.set_ylabel('Customer Count')
    ax2.set_title('Anomaly Consensus Tier\nDistribution')
    for i, (tier, count) in enumerate(tier_counts.items()):
        ax2.text(i, count + 5, str(count),
                 ha='center', fontsize=9)

    # Panel 3 — Isolation Forest score vs return rate
    ax3 = fig.add_subplot(gs[1, 0])
    scatter_colors = df['iso_flagged'].map(
        {True: 'red', False: 'steelblue'}
    )
    ax3.scatter(df['return_rate'], df['iso_anomaly_score'],
                c=scatter_colors, alpha=0.4, s=15)
    ax3.axhline(0, color='black', linestyle='--',
                linewidth=1, label='Decision boundary')
    ax3.set_xlabel('Return Rate')
    ax3.set_ylabel('Anomaly Score (lower = more anomalous)')
    ax3.set_title('Isolation Forest Score\nvs Return Rate')
    ax3.legend(fontsize=8)

    # Panel 4 — method overlap
    ax4 = fig.add_subplot(gs[1, 1])
    overlap = pd.DataFrame({
        'Rule':      df['rule_flagged'].astype(int),
        'SPC':       df['spc_flagged'].astype(int),
        'IsoForest': df['iso_flagged'].astype(int)
    })
    overlap_corr = overlap.corr()
    im = ax4.imshow(overlap_corr, cmap='Blues', vmin=0, vmax=1)
    ax4.set_xticks(range(3))
    ax4.set_yticks(range(3))
    ax4.set_xticklabels(['Rule', 'SPC', 'IsoForest'])
    ax4.set_yticklabels(['Rule', 'SPC', 'IsoForest'])
    for i in range(3):
        for j in range(3):
            ax4.text(j, i, f"{overlap_corr.iloc[i, j]:.2f}",
                     ha='center', va='center', fontsize=10)
    ax4.set_title('Method Agreement\n(Flag Correlation)')
    plt.colorbar(im, ax=ax4, fraction=0.046)

    plt.suptitle('Anomaly Detection — Overview', fontsize=13, y=1.01)
    if save:
        save_figure('anomaly_overview')

def plot_high_confidence_flags(df, save=True):
    """
    Profile chart for high-confidence flagged customers
    vs clean population across key metrics.
    """
    high  = df[df['methods_flagged'] == 3]
    clean = df[df['methods_flagged'] == 0]

    if len(high) == 0:
        print("No high-confidence flags to plot.")
        return

    metrics = ['return_rate', 'avg_order_value',
               'order_velocity', 'avg_discount_pct']
    labels  = ['Return Rate', 'Avg Order Value',
               'Order Velocity', 'Avg Discount %']

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, metric, label in zip(axes, metrics, labels):
        ax.boxplot(
            [clean[metric].dropna(), high[metric].dropna()],
            tick_labels=['Clean', 'High-flag'],
            patch_artist=True,
            boxprops=dict(facecolor='steelblue', alpha=0.6),
        )
        ax.set_title(label)
        ax.set_ylabel(label)

    plt.suptitle(
        'High-Confidence Flags vs Clean Population',
        fontsize=12
    )
    plt.tight_layout()
    if save:
        save_figure('anomaly_high_confidence_profile')


# ── pipeline ──────────────────────────────────────────────────────────────────

def run_anomaly_detection(strictness='moderate',
                          z_threshold=2.5,
                          contamination=0.05):
    """
    Run full anomaly detection pipeline.

    Parameters
    ----------
    strictness :    str   rule-based threshold level
                          ('loose', 'moderate', 'strict')
    z_threshold :   float Z-score cutoff for SPC flags
    contamination : float assumed anomaly proportion for IsoForest

    Returns
    -------
    df : DataFrame  customer-level dataframe with all flag columns
    """
    print("=== ANOMALY DETECTION PIPELINE ===")
    print(f"\nEpistemic note: operating on synthetic data.")
    print(f"Findings demonstrate methodology, not confirmed fraud signals.")
    print(f"Thresholds calibrated from population distribution —")
    print(f"in production, validate against confirmed fraud base rates.\n")

    df        = load_anomaly_data()
    baselines = compute_baselines(df)

    df = rule_based_flags(df, baselines, strictness=strictness)
    df = statistical_process_control(df, baselines,
                                     z_threshold=z_threshold)
    df = isolation_forest_flags(df, contamination=contamination)
    df = consensus_analysis(df)
    tradeoff_analysis(df, baselines)

    plot_anomaly_overview(df, baselines)
    plot_high_confidence_flags(df)

    # Export flagged customer list
    flagged = df[df['methods_flagged'] >= 2][[
        'customer_id', 'country', 'membership_tier',
        'history_depth',
        'total_orders', 'total_returns', 'return_rate',
        'avg_order_value', 'order_velocity',
        'refund_velocity', 'payment_velocity',
        'rule_flag_count', 'spc_flag_count',
        'iso_flagged', 'methods_flagged',
        'consensus_tier', 'iso_anomaly_score'
    ]].sort_values(['methods_flagged', 'iso_anomaly_score'], ascending=[False, True])

    print(f"\n=== FLAGGED CUSTOMER LIST (2+ methods) ===")
    print(f"  {len(flagged)} customers exported for review")
    print(flagged.head(20).to_string(index=False))

    return df, flagged


if __name__ == "__main__":
    df, flagged = run_anomaly_detection()