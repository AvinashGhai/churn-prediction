# Telecom Customer Churn Analysis

An end-to-end data analytics project that predicts customer churn for a telecom company, identifies high-risk customers, and generates actionable retention recommendations using Machine Learning.

<br>

## Live Dashboard
🔗 [View Interactive Tableau Dashboard](https://public.tableau.com/app/profile/avinash.ghai/viz/TelecomCustomerChurnAnalysis_17824221845390/Dashboard1)

<br>

## Project Summary

Analyzed 7,032 telecom customer records to uncover churn patterns, build a predictive ML model, and deliver a business-ready retention strategy — including priority scores and personalized offers for high-risk customers.

<br>

## Key Business Insights

| Finding | Value |
|---------|-------|
| Overall churn rate | 26.5% |
| Month-to-month contract churn | 42.7% |
| Two-year contract churn | 2.8% |
| Highest risk segment | Fiber optic + Month-to-month (54.6%) |
| New customer churn (0–12 months) | ~48% |
| Monthly revenue at risk | $139,131 |

<br>

## What Makes This Project Different

Beyond standard churn prediction, this project includes:

- **Retention Priority Score** — ranks customers by financial risk (`churn_probability × monthly_charges`), not just churn likelihood
- **Next Best Offer Engine** — rule-based system that recommends a personalized retention action for each high-risk customer
- **Top 100 Customers to Save** — exported as an actionable CSV with risk reasons, recommended offers, and priority scores
- **Revenue at Risk Analysis** — quantifies the exact monthly revenue impact of churn

<br>

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest) |
| Database | SQLite |
| Visualization | Tableau Public |
| Version Control | Git, GitHub |

<br>

## Project Structure

```
churn-prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (7,032 customers)
│
├── phase_1.py          # Data cleaning, feature engineering, SQLite storage
├── phase_2.py          # Exploratory data analysis & visualizations
├── phase_3.py          # Random Forest model training & evaluation
├── phase_4_tableau.py  # Export CSVs + retention scoring + offer engine
│
├── sql_queries.sql     # Business SQL queries
│
├── tableau_data/
│   ├── 01_customers_main.csv        # Full customer dataset with risk scores
│   ├── 02_contract_summary.csv      # Churn by contract type
│   ├── 03_internet_summary.csv      # Churn by internet service
│   ├── 04_payment_summary.csv       # Churn by payment method
│   ├── 05_tenure_summary.csv        # Churn by tenure group
│   ├── 06_segment_heatmap.csv       # Contract × internet heatmap
│   ├── 07_predictions.csv           # ML churn probability per customer
│   ├── 08_revenue_summary.csv       # Revenue at risk
│   └── 09_top_customers_to_save.csv # Top 100 priority customers + offers
│
└── README.md
```

<br>

## How to Run

```bash
# Clone the repo
git clone https://github.com/AvinashGhai/churn-prediction.git
cd churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run phases in order
python phase_1.py
python phase_2.py
python phase_3.py
python phase_4_tableau.py
```

<br>

## Dashboard Overview

The Tableau dashboard includes 6 views:

- **Churn by Contract** — bar chart showing churn rate across contract types
- **Churn by Tenure** — churn drops sharply after 12 months
- **Segment Heatmap** — contract × internet service risk matrix
- **Churn Probability Distribution** — histogram of ML risk scores
- **Revenue at Risk** — retained vs churned revenue comparison
- **Top Customers to Save** — ranked list with retention offers

<br>

## ML Model

- **Algorithm:** Random Forest Classifier
- **Dataset:** 7,032 customers, 19 features
- **Key features:** Contract type, tenure, internet service, payment method, monthly charges, tech support

<br>

## Author

**Avinash Ghai**

[![GitHub](https://img.shields.io/badge/GitHub-AvinashGhai-181717?style=flat&logo=github)](https://github.com/AvinashGhai)
[![Tableau](https://img.shields.io/badge/Tableau-Dashboard-E97627?style=flat&logo=tableau&logoColor=white)](https://public.tableau.com/app/profile/avinash.ghai)