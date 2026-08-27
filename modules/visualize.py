# modules/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from modules.config import FIGURES

def save_figure(filename, dpi=150, fmt='png'):
    """Save current figure to outputs/figures with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filepath  = FIGURES / f"{filename}_{timestamp}.{fmt}"
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()  # close without displaying
    print(f"Saved: {filepath}")
    return filepath

def plot_rank_gap(df, save=True):
    """Plot orders rank vs revenue rank gap distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['rank_gap'], bins=20, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(df['rank_gap'].mean(), color='coral',
               linestyle='--', label=f"Mean: {df['rank_gap'].mean():.1f}")
    ax.set_xlabel('Rank Gap (Orders vs Revenue)')
    ax.set_ylabel('Product Count')
    ax.set_title('Distribution of Rank Gaps — Orders vs Revenue')
    ax.legend()
    plt.tight_layout()
    if save:
        save_figure('rank_gap_distribution')
    plt.show()

def plot_discount_vs_orders(df, save=True):
    """Scatterplot of avg discount % vs total orders by category."""
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = df['category'].unique()
    colors = plt.cm.tab20.colors
    for i, cat in enumerate(categories):
        subset = df[df['category'] == cat]
        ax.scatter(subset['avg_discount_pct'], subset['total_orders'],
                   label=cat, color=colors[i % len(colors)], alpha=0.7)
    ax.set_xlabel('Average Discount %')
    ax.set_ylabel('Total Orders')
    ax.set_title('Discount vs Order Volume by Product')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    if save:
        save_figure('discount_vs_orders')
    plt.show()

def plot_before_after_scenario(summary_filtered, recovery_factor=1.15, save=True):
    """Bar chart comparing actual vs simulated orders and revenue."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for ax, actual_col, sim_col, ylabel, title in [
        (axes[0], 'actual_orders',  'simulated_orders',
         'Total Orders',    'Orders: Before vs After'),
        (axes[1], 'actual_revenue', 'simulated_revenue',
         'Revenue (USD)',   'Revenue: Before vs After')
    ]:
        actual = summary_filtered[actual_col].values[0]
        simulated = summary_filtered[sim_col].values[0]
        delta = simulated - actual

        ax.bar(['Actual', 'Simulated'], [actual, simulated],
               color=['steelblue', 'coral'], alpha=0.8, width=0.4)
        ax.annotate(
            f'+{delta:,.0f}' if delta >= 0 else f'{delta:,.0f}',
            xy=(1, simulated),
            xytext=(0.5, simulated * 0.95),
            ha='center', color='darkred', fontsize=10
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    plt.suptitle(
        f'Scenario Model: Discount Removal\n({int((recovery_factor-1)*100)}% Recovery Assumption)',
        fontsize=11
    )
    plt.tight_layout()
    if save:
        save_figure('before_after_scenario')
    plt.show()

def plot_country_profile(df, country, save=True):
    """Grouped bar chart of spend by age group and gender for one country."""
    country_df = df[df['country'] == country].copy()
    if len(country_df) == 0:
        print(f"No data for {country}")
        return

    age_order = ['18-24', '25-31', '32-38', '39-45', '46-52', '53-59', '60+']
    genders   = ['Female', 'Male', 'Other']
    colors    = {'Female': 'steelblue', 'Male': 'coral', 'Other': 'mediumseagreen'}
    width     = 0.25
    offsets   = [-width, 0, width]
    x         = range(len(age_order))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, label in [
        (axes[0], 'total_spend', 'Total Spend (USD)'),
        (axes[1], 'avg_spend',   'Avg Spend per Customer (USD)')
    ]:
        for i, gender in enumerate(genders):
            subset = country_df[country_df['gender'] == gender].set_index('age_group')
            values = [subset.loc[age, metric] if age in subset.index else 0
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

    plt.suptitle(f'{country} Customer Profile by Age Group and Gender', fontsize=12)
    plt.tight_layout()
    if save:
        filename = f"profile_{country.lower().replace(' ', '_')}"
        save_figure(filename)
    plt.show()

def plot_conversion_heatmap(df, top_n=8, mode='indexed', save=True):
    """
    Heatmap of revenue by country and age group.
    mode: 'split' separates US; 'indexed' normalizes per country; 'log' uses log scale.
    """
    age_order = ['18-24', '25-31', '32-38', '39-45', '46-52', '53-59', '60+']

    country_totals = df.groupby('country')['total_spend'].sum().sort_values(ascending=False)
    top_countries  = country_totals.head(top_n).index.tolist()
    top_df = df[df['country'].isin(top_countries)].groupby(
        ['country', 'age_group']
    )['total_spend'].sum().reset_index()

    pivot = top_df.pivot(
        index='country', columns='age_group', values='total_spend'
    ).reindex(index=top_countries, columns=age_order)

    if mode == 'indexed':
        plot_data = pivot.div(pivot.max(axis=1), axis=0) * 100
        title     = 'Age Group Concentration by Market (Index: 100 = peak)'
        cbar_label = 'Index (100 = top age group per country)'
        annot      = pivot.values
        fmt        = '.0f'
        vmin, vmax = 0, 100
        norm       = None
    elif mode == 'log':
        plot_data  = pivot
        title      = 'Revenue by Market and Age Group (Log Scale)'
        cbar_label = 'Total Spend (USD) — log scale'
        annot      = True
        fmt        = '.0f'
        vmin = vmax = None
        norm = plt.matplotlib.colors.LogNorm(
            vmin=pivot.min().min(), vmax=pivot.max().max()
        )
    else:
        plot_data  = pivot
        title      = 'Revenue by Market and Age Group'
        cbar_label = 'Total Spend (USD)'
        annot      = True
        fmt        = '.0f'
        vmin = vmax = None
        norm        = None

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