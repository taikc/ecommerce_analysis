# E-Commerce Analysis Pipeline

End-to-end analytical pipeline examining customer behavior, product 
performance, and revenue patterns across a synthetic e-commerce dataset 
(2020–2026). Built to demonstrate analytical reasoning, modular Python 
architecture, and SQL proficiency.

---

## Project Structure
ecommerce_analysis/
├── modules/
│ ├── config.py # paths and environment settings
│ ├── db.py # database connection and query helpers
│ ├── validate.py # data quality checks
│ ├── products.py # product performance analysis
│ ├── customers.py # customer segmentation and churn
│ ├── ltv.py # lifetime value pipeline
│ ├── orders.py # order-level behavioral analysis
│ ├── exports.py # XLSX and snapshot exports
│ └── visualize.py # plotting utilities
├── outputs/
│ ├── figures/ # generated charts (gitignored)
│ └── reports/ # data quality reports (gitignored)
├── data/
│ ├── raw/ # source CSVs (gitignored)
│ └── exports/ # Tableau-ready XLSX (gitignored)
├── main.py # pipeline orchestrator
├── requirements.txt
└── README.md

---

## Business Questions

**Products**
- Does order volume predict revenue, or does price tier explain divergence?
- Does discounting drive order volume, or compress margin without demand impact?
- Which products are structurally viable across quality, return rate, and revenue?

**Customers**
- Which market, age, and gender segments generate the highest lifetime value (LTV)?
- Does membership tier correlate with retention quality?
- What behavioral signals predict churn most reliably?

**Orders**
- Which temporal patterns drive revenue concentration?
- Do session depth and engagement predict order value?
- Are return rate differences across categories statistically significant?

---

## Key Findings

Full analytical narrative and methodology available in the 
[Kaggle notebook]([link](https://www.kaggle.com/code/taikchiba/e-commerce-data-analysis)).

Selected highlights:
- Discounting shows no significant demand effect across 13 of 14 
  categories — the exception (Jewelry) shows a negative correlation
- Gold tier customers churn at higher rates than Free tier despite 
  34% higher average spend
- Recency dominates churn prediction; satisfaction signals add 
  marginal independent value

---

## Dataset

Synthetic e-commerce dataset comprising four tables:

| Table |      Rows |     Grain |
| `product_summary` | 140 | One row per product |
| `customers` | 8,000 | One row per customer |
| `orders` | 25,000 | One row per transaction |
| `monthly_revenue` | 75 | One row per month |

Data is not included in this repository. To reproduce:
1. Download the dataset from Kaggle [link][(https://www.kaggle.com/datasets/meruvakodandasuraj/e-commerce-customer-behavior-and-sales-20202026/)]
Or via kagglehub ("meruvakodandasuraj/e-commerce-customer-behavior-and-sales-20202026")
2. Place CSV files in `data/raw/`
3. Run `python setup_db.py` to build the SQLite database

---

## Setup

```bash
# Clone the repository
git clone https://github.com/taikc/ecommerce_analysis.git
cd ecommerce_analysis

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate       # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Build database from CSVs (after placing files in data/raw/)
python setup_db.py
```

---

## Usage

```bash
# Full pipeline
python main.py

# Single module
python main.py --module products
python main.py --module customers
python main.py --module orders
python main.py --module validate

# Skip stages
python main.py --skip-validate
python main.py --skip-exports
```

---

## Technical Notes

- SQLite used as single database source of truth, shared between DBeaver 
  (SQL exploration) and Python (analysis pipeline)
- Statistical tests selected by data type: Pearson for continuous, 
  Mann-Whitney U for non-normal distributions, chi-square for binary outcomes, 
  Kruskal-Wallis for multi-group comparisons
- LTV projection discounted by recency-derived churn probability rather than 
  binary churn flag — active customers with high recency receive proportionally 
  lower forward projections
- Class imbalance in churn model (91/9 split) addressed with 
  `class_weight='balanced'`, improving churned customer recall from 0.02 to 0.62

---

## Tools

Python · SQLite · DBeaver · pandas · matplotlib · seaborn · 
scipy · scikit-learn · statsmodels · Git

---

## Related

- [Kaggle Notebook — [link](https://www.kaggle.com/code/taikchiba/e-commerce-data-analysis)] Analytical narrative with methodology and findings
- [Underville — [link](https://medium.com/@underville)] Critical writing on analytical practice
