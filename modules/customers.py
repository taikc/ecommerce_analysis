# modules/customers.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modules.db import get_connection
from modules.visualize import save_figure

def load_customer_data(conn=None):
    """Load full customer table from database."""
    if conn is None:
        conn = get_connection()
    return pd.read_sql("SELECT * FROM customers", conn)

def add_age_group(df):
    """Bin age into 7-year ranges starting at 18."""
    df = df.copy()
    df['age_group'] = pd.cut(
        df['age'],
        bins=[17, 24, 31, 38, 45, 52, 59, 100],
        labels=['18-24','25-31','32-38','39-45','46-52','53-59','60+']
    )
    return df

def profile_overview(df):
    """Print headline customer metrics."""
    total     = len(df)
    churned   = df['churned'].sum()
    print("=== CUSTOMER PROFILE OVERVIEW ===")
    print(f"  Total customers:       {total:,}")
    print(f"  Countries:             {df['country'].nunique()}")
    print(f"  Avg age:               {df['age'].mean():.1f}")
    print(f"  Avg orders/customer:   {df['total_orders'].mean():.1f}")
    print(f"  Avg spend/customer:    ${df['total_spend_usd'].mean():,.2f}")
    print(f"  Avg order value:       ${df['avg_order_value_usd'].mean():,.2f}")
    print(f"  Avg recency (days):    {df['days_since_last_purchase'].mean():.1f}")
    print(f"  Churn rate:            {churned / total * 100:.1f}%")

def churn_by_cohort(df):
    """
    Churn rate by registration year cohort.
    Pre-2020 customers are grouped into a single bucket
    due to small sample sizes producing noisy rates.
    """
    df = df.copy()
    df['registration_year'] = pd.to_datetime(df['registration_date']).dt.year
    df['cohort'] = df['registration_year'].apply(
        lambda y: 'pre_2020' if y < 2020 else str(y)
    )
    result = df.groupby('cohort').agg(
        total_customers =('customer_id',          'count'),
        churned         =('churned',               'sum'),
        avg_spend       =('total_spend_usd',       'mean'),
        avg_recency_days=('days_since_last_purchase','mean')
    ).round(2)
    result['churn_rate_pct'] = (
        result['churned'] / result['total_customers'] * 100
    ).round(1)
    return result

def membership_tier_profile(df):
    """
    Aggregate customer metrics by membership tier.
    Highlights Gold tier mid-trap dynamic:
    high avg spend but highest churn rate.
    """
    result = df.groupby('membership_tier').agg(
        customer_count  =('customer_id',            'count'),
        avg_spend       =('total_spend_usd',         'mean'),
        avg_order_value =('avg_order_value_usd',     'mean'),
        avg_recency_days=('days_since_last_purchase','mean'),
        churned         =('churned',                 'sum')
    ).round(2)
    result['pct_of_total'] = (
        result['customer_count'] / result['customer_count'].sum() * 100
    ).round(1)
    result['churn_rate_pct'] = (
        result['churned'] / result['customer_count'] * 100
    ).round(1)
    result['value_retention_ratio'] = (
        result['avg_spend'] / result['churn_rate_pct']
    ).round(0)
    return result.reindex(['Free', 'Silver', 'Gold', 'Platinum'])

def gold_churn_scenarios(tier_df):
    """
    Scenario model: recoverable revenue if Gold tier churn
    is reduced to various target rates.
    Silver's rate (7.1%) used as the primary benchmark.
    """
    gold = tier_df.loc['Gold']
    scenarios = [
        ('Conservative', 9.0),
        ('Moderate',     8.0),
        ('Aggressive',   7.1),
        ('Best case',    6.0),
    ]
    print(f"\n{'Scenario':<15} {'Target Churn':>13} {'Recoverable Revenue':>20}")
    print("-" * 50)
    rows = []
    for label, target in scenarios:
        saved     = gold['customer_count'] * (gold['churn_rate_pct'] - target) / 100
        recovered = saved * gold['avg_spend']
        print(f"{label:<15} {target:>12.1f}% ${recovered:>18,.2f}")
        rows.append({'scenario': label, 'target_churn': target,
                     'recoverable_revenue': round(recovered, 2)})
    return pd.DataFrame(rows)

def audience_conversion(df):
    """
    Aggregate total and avg spend by country, age group and gender.
    Returns sorted by total spend descending.
    """
    df = add_age_group(df)
    result = df.groupby(
        ['country', 'age_group', 'gender'], observed=True
    ).agg(
        customer_count =('customer_id',         'count'),
        avg_spend      =('total_spend_usd',      'mean'),
        avg_order_value=('avg_order_value_usd',  'mean'),
        total_spend    =('total_spend_usd',       'sum')
    ).round(2).reset_index()
    return result.sort_values('total_spend', ascending=False)

def country_summary(conversion_df):
    """Aggregate conversion data to country level."""
    return conversion_df.groupby('country').agg(
        total_customers=('customer_count', 'sum'),
        total_revenue  =('total_spend',    'sum'),
        avg_spend      =('avg_spend',      'mean'),
        avg_order_value=('avg_order_value','mean')
    ).round(2).sort_values('total_revenue', ascending=False)

def plot_conversion_heatmap(conversion_df, top_n=8, mode='indexed', save=True):
    """
    Heatmap of revenue concentration by country and age group.
    mode='indexed': normalizes each country to its own peak (default)
    mode='log':     log scale, all countries in one chart
    """
    age_order      = ['18-24','25-31','32-38','39-45','46-52','53-59','60+']
    country_totals = conversion_df.groupby('country')['total_spend'].sum()\
                                  .sort_values(ascending=False)
    top_countries  = country_totals.head(top_n).index.tolist()

    top_df = conversion_df[conversion_df['country'].isin(top_countries)]\
        .groupby(['country','age_group'], observed=True)['total_spend']\
        .sum().reset_index()

    pivot = top_df.pivot(
        index='country', columns='age_group', values='total_spend'
    ).reindex(index=top_countries, columns=age_order)

    if mode == 'indexed':
        plot_data  = pivot.div(pivot.max(axis=1), axis=0) * 100
        title      = 'Age Group Concentration by Market (Index: 100 = peak)'
        cbar_label = 'Index (100 = top age group per country)'
        annot, fmt = pivot.values, '.0f'
        vmin, vmax, norm = 0, 100, None
    else:
        plot_data  = pivot
        title      = 'Revenue by Market and Age Group (Log Scale)'
        cbar_label = 'Total Spend (USD) — log scale'
        annot, fmt = True, '.0f'
        vmin = vmax = None
        norm = plt.matplotlib.colors.LogNorm(
            vmin=pivot.min().min(), vmax=pivot.max().max()
        )

    plt.figure(figsize=(13, 7))
    sns.heatmap(plot_data, annot=annot, fmt=fmt, cmap='YlOrRd',
                linewidths=0.5, vmin=vmin, vmax=vmax, norm=norm,
                cbar_kws={'label': cbar_label})
    plt.title(title)
    plt.xlabel('Age Group')
    plt.ylabel('Country')
    plt.tight_layout()
    if save:
        save_figure(f'conversion_heatmap_{mode}')
    plt.show()

def plot_country_profile(conversion_df, country, save=True):
    """
    Grouped bar chart of total and avg spend by age group and gender
    for a single country.
    """
    country_df = conversion_df[conversion_df['country'] == country].copy()
    if len(country_df) == 0:
        print(f"No data for {country}")
        return

    age_order = ['18-24','25-31','32-38','39-45','46-52','53-59','60+']
    genders   = ['Female', 'Male', 'Other']
    colors    = {'Female': 'steelblue', 'Male': 'coral', 'Other': 'mediumseagreen'}
    width, offsets = 0.25, [-0.25, 0, 0.25]
    x = range(len(age_order))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, label in [
        (axes[0], 'total_spend', 'Total Spend (USD)'),
        (axes[1], 'avg_spend',   'Avg Spend per Customer (USD)')
    ]:
        for i, gender in enumerate(genders):
            subset = country_df[country_df['gender'] == gender]\
                        .set_index('age_group')
            values = [subset.loc[age, metric]
                      if age in subset.index else 0
                      for age in age_order]
            ax.bar([xi + offsets[i] for xi in x], values,
                   width=width, label=gender,
                   color=colors[gender], alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(age_order)
        ax.set_xlabel('Age Group')
        ax.set_ylabel(label)
        ax.set_title(f'{country} — {label}')
        ax.legend()

    plt.suptitle(f'{country} Customer Profile by Age Group and Gender',
                 fontsize=12)
    plt.tight_layout()
    if save:
        save_figure(f"profile_{country.lower().replace(' ', '_')}")
    plt.show()

def product_satisfaction_analysis(df):
    """
    Analyze correlation among reviews_given, avg_review_score, returns_made.
    Aggregates satisfaction metrics by key cohort dimensions.
    """
    import scipy.stats as stats

    satisfaction_cols = ['reviews_given', 'avg_review_score', 'returns_made']

    # Step 1 — correlation matrix among satisfaction metrics
    corr_matrix = df[satisfaction_cols].corr(method='pearson').round(3)
    print("=== SATISFACTION METRIC CORRELATIONS ===")
    print(corr_matrix.to_string())

    # Step 2 — significance tests for each pair
    print("\n=== PAIRWISE SIGNIFICANCE TESTS ===")
    pairs = [
        ('reviews_given',   'avg_review_score'),
        ('reviews_given',   'returns_made'),
        ('avg_review_score','returns_made')
    ]
    for col_a, col_b in pairs:
        r, p = stats.pearsonr(df[col_a].dropna(), df[col_b].dropna())
        sig  = "significant" if p < 0.05 else "not significant"
        print(f"  {col_a:20s} vs {col_b:20s} r={r:+.3f}  p={p:.4f}  ({sig})")

    # Step 3 — satisfaction by cohort dimensions
    print("\n=== SATISFACTION BY MEMBERSHIP TIER ===")
    tier_sat = df.groupby('membership_tier')[satisfaction_cols].mean().round(3)
    print(tier_sat.reindex(['Free','Silver','Gold','Platinum']).to_string())

    print("\n=== SATISFACTION BY PREFERRED CATEGORY ===")
    cat_sat = df.groupby('preferred_category')[satisfaction_cols]\
                .mean().round(3).sort_values('avg_review_score', ascending=False)
    print(cat_sat.to_string())

    print("\n=== SATISFACTION BY ACQUISITION CHANNEL ===")
    channel_sat = df.groupby('acquisition_channel')[satisfaction_cols]\
                    .mean().round(3).sort_values('avg_review_score', ascending=False)
    print(channel_sat.to_string())

    return {
        'correlation_matrix': corr_matrix,
        'by_tier':            tier_sat,
        'by_category':        cat_sat,
        'by_channel':         channel_sat
    }

def churn_prediction_model(df):
    """
    Logistic regression predicting churn from satisfaction
    and activity signals. Returns coefficients and model accuracy.
    Connects descriptive churn analysis to a predictive layer.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing  import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics         import classification_report

    features = [
        'avg_review_score', 'reviews_given', 'returns_made',
        'wishlist_items',   'newsletter_subscribed',
        'days_since_last_purchase', 'total_orders',
        'avg_order_value_usd'
    ]

    model_df = df[features + ['churned']].dropna()
    X = model_df[features]
    y = model_df['churned']

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Weighting compensates for 91%-nonchurn/9%-churn imbalance
    model = LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    )
    model.fit(X_train, y_train)

    print("=== CHURN PREDICTION MODEL ===")
    print(classification_report(y_test, model.predict(X_test)))

    # Feature importance via coefficients
    coef_df = pd.DataFrame({
        'feature':     features,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)

    print("\nFeature coefficients (strongest churn predictors):")
    print(coef_df.to_string(index=False))

    return model, coef_df

def customer_activity_analysis(df):
    """
    Correlation between engagement signals and purchase behavior.
    Hypothesis: low signal expected due to random engagement patterns.
    """
    import scipy.stats as stats

    engagement_cols = ['reviews_given', 'wishlist_items', 'newsletter_subscribed']
    purchase_cols   = ['total_orders', 'total_spend_usd', 'avg_order_value_usd']

    print("=== ENGAGEMENT vs PURCHASE BEHAVIOR ===")
    print(f"{'Engagement':<25} {'Purchase metric':<25} {'r':>8} {'p':>8} {'sig':>16}")
    print("-" * 85)

    results = []
    for eng in engagement_cols:
        for pur in purchase_cols:
            r, p  = stats.pearsonr(df[eng], df[pur])
            sig   = "significant" if p < 0.05 else "not significant"
            print(f"{eng:<25} {pur:<25} {r:>+8.3f} {p:>8.4f} {sig:>16}")
            results.append({
                'engagement': eng, 'purchase': pur,
                'r': round(r, 3), 'p': round(p, 4),
                'significant': p < 0.05
            })

    return pd.DataFrame(results)

def acquisition_channel_analysis(df):
    """
    Compare customer quality metrics across acquisition channels.
    Uses Kruskal-Wallis test to assess statistical significance
    of spend differences across channels.
    """
    import scipy.stats as stats
    from scipy.stats import chi2_contingency

    metrics = [
        'total_spend_usd', 'avg_order_value_usd',
        'total_orders',    'days_since_last_purchase'
    ]

    # Descriptive aggregation
    channel_df = df.groupby('acquisition_channel').agg(
        customer_count  =('customer_id',            'count'),
        avg_spend       =('total_spend_usd',         'mean'),
        avg_order_value =('avg_order_value_usd',     'mean'),
        avg_orders      =('total_orders',            'mean'),
        avg_recency     =('days_since_last_purchase','mean'),
        churn_rate      =('churned',                 'mean')
    ).round(2)
    channel_df['churn_rate'] = (channel_df['churn_rate'] * 100).round(1)
    channel_df = channel_df.sort_values('avg_spend', ascending=False)

    print("=== ACQUISITION CHANNEL PROFILE ===")
    print(channel_df.to_string())

    # Kruskal-Wallis test — are spend differences significant?
    print("\n=== KRUSKAL-WALLIS TEST (spend by channel) ===")
    groups = [
        group['total_spend_usd'].values
        for _, group in df.groupby('acquisition_channel')
    ]
    stat, p = stats.kruskal(*groups)
    sig = "significant" if p < 0.05 else "not significant"
    print(f"  H-statistic: {stat:.3f}  p={p:.4f}  ({sig})")
    print(f"  Interpretation: channel differences in spend are {sig}.")

    contingency = pd.crosstab(df['acquisition_channel'], df['churned'])
    chi2, p, dof, _ = chi2_contingency(contingency)
    sig = "significant" if p < 0.05 else "not significant"
    print(f"\nChi-square test (churn by channel):")
    print(f"  chi2={chi2:.3f}  p={p:.4f}  ({sig})")

    return channel_df

def run_extended_customer_analysis(df=None):
    """Run satisfaction, activity and channel analyses."""
    if df is None:
        df = load_customer_data()

    print("\n" + "="*50)
    sat  = product_satisfaction_analysis(df)

    print("\n" + "="*50)
    act  = customer_activity_analysis(df)

    print("\n" + "="*50)
    chan = acquisition_channel_analysis(df)

    print("\n" + "="*50)
    print("=== CHURN PREDICTION ===")
    model, coefs = churn_prediction_model(df)

    return {
        'satisfaction': sat,
        'activity':     act,
        'channel':      chan,
        'churn_model':  coefs
    }

def run_customer_analysis():
    """Run full customer analysis pipeline."""
    df = load_customer_data()
    df = add_age_group(df)

    profile_overview(df)

    print("\n=== CHURN BY COHORT ===")
    cohort = churn_by_cohort(df)
    print(cohort.to_string())

    print("\n=== MEMBERSHIP TIER PROFILE ===")
    tiers = membership_tier_profile(df)
    print(tiers.to_string())

    print("\n=== GOLD TIER CHURN SCENARIOS ===")
    scenarios = gold_churn_scenarios(tiers)

    print("\n=== AUDIENCE CONVERSION (top 20 segments) ===")
    conversion = audience_conversion(df)
    print(conversion.head(20).to_string(index=False))

    print("\n=== COUNTRY SUMMARY ===")
    c_summary = country_summary(conversion)
    print(c_summary.to_string())

    return {
        'data':       df,
        'cohort':     cohort,
        'tiers':      tiers,
        'scenarios':  scenarios,
        'conversion': conversion,
        'country':    c_summary
    }

if __name__ == "__main__":
    results = run_customer_analysis()
    run_extended_customer_analysis()