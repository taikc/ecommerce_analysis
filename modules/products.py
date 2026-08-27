# modules/products.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from modules.db import get_connection, query
from modules.visualize import save_figure

def load_product_data(conn=None):
    """Load product summary table from database."""
    if conn is None:
        conn = get_connection()
    return pd.read_sql("SELECT * FROM product_summary", conn)

def rank_gap_analysis(df):
    """
    Identify products where orders rank and revenue rank diverge significantly.
    Uses standard deviation as threshold — statistically grounded, not arbitrary.
    """
    df = df.copy()
    df['orders_rank']  = df['total_orders'].rank(ascending=False)
    df['revenue_rank'] = df['total_revenue_usd'].rank(ascending=False)
    df['rank_gap']     = (df['orders_rank'] - df['revenue_rank']).abs()

    avg_gap = df['rank_gap'].mean()
    stdev   = df['rank_gap'].std()
    threshold = avg_gap + stdev

    df['flagged'] = df['rank_gap'] > threshold

    print(f"Rank gap — mean: {avg_gap:.2f}, stdev: {stdev:.2f}, threshold: {threshold:.2f}")
    print(f"Flagged products: {df['flagged'].sum()} of {len(df)}")

    return df

def abc_classification(df):
    """
    Classify products into ABC tiers by cumulative revenue contribution.
    A: top 80%, B: next 15%, C: bottom 5%.
    """
    df = df.copy().sort_values('total_revenue_usd', ascending=False)
    grand_total    = df['total_revenue_usd'].sum()
    df['cum_pct']  = df['total_revenue_usd'].cumsum() / grand_total * 100
    df['abc_class'] = df['cum_pct'].apply(
        lambda x: 'A' if x <= 80 else ('B' if x <= 95 else 'C')
    )
    return df

def viability_score(df):
    """
    Composite product viability score combining rating, return rate,
    revenue contribution and delivery speed.
    Weights: rating 30%, return rate 30%, revenue 25%, delivery 15%.
    """
    df = df.copy()
    df['viability_score'] = (
        (df['avg_rating'] / 5.0) * 0.30 +
        (1 - df['return_rate'] / 100.0) * 0.30 +
        (df['total_revenue_usd'] / df['total_revenue_usd'].max()) * 0.25 +
        (1 - df['avg_delivery_days'] / df['avg_delivery_days'].max()) * 0.15
    ).round(3)
    return df

def abc_viability_matrix(df):
    """
    Combine ABC class and viability score into actionable recommendations.
    Returns product-level dataframe with recommendation labels.
    """
    df = abc_classification(df)
    df = viability_score(df)

    conditions = [
        (df['abc_class'] == 'A') & (df['viability_score'] >= 0.7),
        (df['abc_class'] == 'A') & (df['viability_score'] <  0.7),
        (df['abc_class'] == 'B') & (df['viability_score'] >= 0.6),
        (df['abc_class'] == 'B') & (df['viability_score'] <  0.6),
        (df['abc_class'] == 'C') & (df['viability_score'] >= 0.6),
    ]
    labels = [
        'Protect',
        'Investigate',
        'Grow',
        'Monitor',
        'Nurture',
    ]

    import numpy as np
    df['recommendation'] = np.select(conditions, labels, default='Review for removal')
    return df[['category', 'product_name', 'total_revenue_usd', 'total_orders',
               'avg_rating', 'return_rate', 'viability_score',
               'abc_class', 'recommendation']].sort_values(
                   ['abc_class', 'viability_score'], ascending=[True, False]
               )

def discount_correlation(df):
    """
    Pearson correlation between avg discount % and total orders per category.
    Flags statistically significant results (p < 0.05).
    """
    print("Pearson correlation: discount % vs total orders\n")
    results = []
    for cat, group in df.groupby('category'):
        if len(group) > 3:
            r, p = stats.pearsonr(group['avg_discount_pct'], group['total_orders'])
            significant = p < 0.05
            significance = "significant" if significant else "not significant"
            print(f"{cat:30s} r={r:+.3f}  p={p:.3f}  ({significance})")
            results.append({
                'category': cat, 'r': r, 'p': p, 'significant': significant
            })
    return pd.DataFrame(results)

def bootstrap_confidence_interval(df, category='Jewelry & Accessories',
                                   n_bootstrap=10000):
    """
    Bootstrap 95% confidence interval for discount/orders correlation
    in a specific category. Translates bounds into revenue uplift estimates.
    """
    subset = df[df['category'] == category].copy()
    np.random.seed(42)
    correlations = []

    for _ in range(n_bootstrap):
        sample = subset.sample(n=len(subset), replace=True)
        r, _   = stats.pearsonr(
            sample['avg_discount_pct'], sample['total_orders']
        )
        correlations.append(r)

    ci_lower   = np.percentile(correlations, 2.5)
    ci_upper   = np.percentile(correlations, 97.5)
    r_observed = subset['avg_discount_pct'].corr(subset['total_orders'])
    avg_price  = subset['avg_price'].mean()
    avg_orders = subset['total_orders'].mean()
    discount_reduction = subset['avg_discount_pct'].mean()

    print(f"\n{category} — Bootstrap CI (n={n_bootstrap})")
    print(f"Observed r:    {r_observed:.3f}")
    print(f"95% CI:        [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"\n{'Scenario':<12} {'r':>8} {'Revenue Uplift':>16}")
    print("-" * 38)

    for label, r in [('Best', ci_lower), ('Expected', r_observed), ('Worst', ci_upper)]:
        order_change   = abs(r) * discount_reduction * avg_orders / 100
        revenue_impact = order_change * avg_price
        print(f"{label:<12} {r:>8.3f} ${revenue_impact:>14,.0f}")

    return {
        'r_observed': r_observed,
        'ci_lower':   ci_lower,
        'ci_upper':   ci_upper,
        'category':   category
    }

def scenario_model(df, category='Jewelry & Accessories', recovery_factor=1.15):
    """
    Simple before/after scenario model for discount removal.
    Projects order and revenue uplift using a tunable recovery factor.
    """
    df = df.copy()
    mask = df['category'] == category

    df['simulated_orders']  = df['total_orders']
    df['simulated_revenue'] = df['total_revenue_usd']

    df.loc[mask, 'simulated_orders']  = (
        df.loc[mask, 'total_orders'] * recovery_factor
    ).round(0)
    df.loc[mask, 'simulated_revenue'] = (
        df.loc[mask, 'total_revenue_usd'] * recovery_factor
    )

    summary = df.groupby('category').agg(
        actual_orders   =('total_orders',       'sum'),
        simulated_orders=('simulated_orders',   'sum'),
        actual_revenue  =('total_revenue_usd',  'sum'),
        simulated_revenue=('simulated_revenue', 'sum')
    ).round(0)

    summary['revenue_delta'] = (
        summary['simulated_revenue'] - summary['actual_revenue']
    )
    return summary[summary['revenue_delta'] != 0]

def ab_test_feasibility(effect_size_ratio=0.5, power=0.80, alpha=0.05):
    """
    Calculate minimum sample size for a two-group A/B test.
    Default parameters detect a 15% order change with 80% power.
    """
    from statsmodels.stats.power import TTestIndPower
    analysis   = TTestIndPower()
    n_per_group = analysis.solve_power(
        effect_size=effect_size_ratio,
        power=power,
        alpha=alpha
    )
    print(f"A/B test — products needed per group: {n_per_group:.0f}")
    print(f"  Effect size: {effect_size_ratio}, Power: {power}, Alpha: {alpha}")
    return n_per_group

def run_product_analysis():
    """Run full product analysis pipeline."""
    df = load_product_data()

    print("=== RANK GAP ANALYSIS ===")
    df_ranked = rank_gap_analysis(df)

    print("\n=== ABC + VIABILITY MATRIX ===")
    matrix = abc_viability_matrix(df)
    print(matrix.to_string(index=False))

    print("\n=== DISCOUNT CORRELATION ===")
    corr_results = discount_correlation(df)

    print("\n=== BOOTSTRAP CONFIDENCE INTERVAL ===")
    ci = bootstrap_confidence_interval(df)

    print("\n=== SCENARIO MODEL ===")
    scenario = scenario_model(df)
    print(scenario.to_string())

    print("\n=== A/B TEST FEASIBILITY ===")
    ab_test_feasibility()

    return {
        'data':        df_ranked,
        'matrix':      matrix,
        'correlation': corr_results,
        'ci':          ci,
        'scenario':    scenario
    }

if __name__ == "__main__":
    run_product_analysis()