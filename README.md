# Telecom Customer Churn Analysis

End-to-end data analytics project that predicts customer churn for a telecom company using Machine Learning and visualizes insights through an interactive Tableau dashboard.

## Live Dashboard
[View on Tableau Public](https://public.tableau.com/app/profile/avinash.ghai/viz/TelecomCustomerChurnAnalysis_17824221845390/Dashboard1)

## Key Findings

- Month-to-month contract customers churn at **42.7%** vs only **2.8%** for two-year contracts
- New customers (0–12 months) have the highest churn rate at **~48%**
- Fiber optic + month-to-month is the highest risk segment with **54.6% churn rate**
- Revenue at risk from churned customers: **$139,131/month**
- ML model assigns churn probability scores to each customer for proactive intervention

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest) |
| Database | SQLite |
| Visualization | Tableau Public |

## Project Structure

```
churn-prediction/
│
├── churn.db                     # SQLite database
├── model/
│   ├── churn_model.pkl          # Trained ML model
│   └── feature_names.pkl        # Feature list
│
├── tableau_data/                # Exported CSVs for Tableau
│   ├── 01_customers_main.csv
│   ├── 02_contract_summary.csv
│   ├── 03_internet_summary.csv
│   ├── 04_payment_summary.csv
│   ├── 05_tenure_summary.csv
│   ├── 06_segment_heatmap.csv
│   ├── 07_predictions.csv
│   └── 08_revenue_summary.csv
│
├── phase_1.py          # Data cleaning & loading
├── phase_2.py                # Exploratory Data Analysis
├── phase_3.py                 # ML model training
└── phase_4_tableau.py     # Tableau data export
```

## How to Run

```bash
git clone https://github.com/AvinashGhai/churn-prediction.git
cd churn-prediction
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn
python phase1_data_prep.py
python phase2_eda.py
python phase3_ml.py
python phase4_tableau_export.py
```

## Dashboard Includes

- Churn rate by contract type
- Churn rate by customer tenure
- Segment heatmap (contract × internet service)
- Churn probability distribution
- Revenue at risk analysis

## Author

**Avinash Ghai**  
[GitHub](https://github.com/AvinashGhai)
