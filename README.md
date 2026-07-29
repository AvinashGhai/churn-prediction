# Telecom Customer Churn Prediction

An end-to-end data analytics project that identifies at-risk telecom customers using SQL, Python, and Machine Learning, with results visualized in an interactive Tableau dashboard — and personalized retention outreach generated with GenAI.

**Live Dashboard:** https://public.tableau.com/app/profile/avinash.ghai/viz/TelecomCustomerChurnAnalysis_17824221845390/Dashboard1

---

## Dashboard Preview
<img width="730" height="430" alt="image" src="https://github.com/user-attachments/assets/610835e6-1e3d-4026-92b6-910973d25bcf" />




<img width="726" height="424" alt="image" src="https://github.com/user-attachments/assets/d3ad5098-f5ad-4fba-9d7e-1820d43f0263" />


## Business Problem

Customer churn is one of the most critical challenges in the telecom industry. Every customer who cancels their subscription represents lost recurring revenue and acquisition cost. This project identifies which customers are at the highest risk of churning and why, enabling the business to take targeted retention action before it is too late.

---

## Key Findings

| Metric | Value |
|--------|-------|
| Overall churn rate | 26.6% |
| Month-to-month contract churn rate | 42.7% |
| Two-year contract churn rate | 2.8% |
| Fiber optic + month-to-month churn rate | 54.6% |
| Electronic check payment churn rate | 45.3% |
| New customer churn rate (0-12 months) | 47.7% |
| Monthly revenue at risk | $139,131 |
| Recommended outreach threshold | 0.67 |
| High-confidence customers to contact | 129 |
| Estimated net monthly campaign value | $420.81 |

**Top risk factors identified:**
1. Month-to-month contract — customers are 15x more likely to churn than two-year customers
2. Electronic check payment — 3x higher churn than automatic payment methods
3. New customers (under 12 months tenure) — nearly half churn before their first year
4. Fiber optic internet without tech support — high spend combined with low retention

---

## Project Structure

```
churn-prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Raw dataset (Kaggle)
├── churn.db                                 # SQLite database
│
├── phase_1.py                               # Data loading, cleaning, SQL EDA
├── phase_2.py                               # Python EDA and visualizations
├── phase_3.py                               # Feature engineering and ML model
├── phase_5_threshold_analysis.py            # Threshold optimization and business impact
├── phase_4_tableau.py                       # Export Tableau-ready data using the chosen threshold
├── phase_6_genai_offers.py                  # GenAI-generated personalized retention emails
│
├── sql_queries.sql                          # Reference SQL queries
├── requirements.txt                         # Python dependencies
│
├── charts/                                  # 10 generated chart PNGs
├── model/
│   ├── churn_model.pkl                      # Trained Random Forest model
│   ├── scaler.pkl                           # Feature scaler
│   ├── feature_names.pkl                    # Column names for inference
│   └── best_threshold.txt                   # Recommended outreach threshold
│
└── tableau_data/                            # CSVs used in Tableau dashboard
    ├── 01_customers_main.csv
    ├── 02_contract_summary.csv
    ├── 03_internet_summary.csv
    ├── 04_payment_summary.csv
    ├── 05_tenure_summary.csv
    ├── 06_segment_heatmap.csv
    ├── 07_predictions.csv
    ├── 08_revenue_summary.csv
    ├── 09_top_customers_to_save.csv
    ├── 10_threshold_comparison.csv
    ├── 11_threshold_full_curve.csv
    └── 12_genai_retention_emails.csv
```

---

## Dataset

- **Source:** IBM Telco Customer Churn via Kaggle
- **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Size:** 7,032 customers x 21 features
- **Target variable:** Churn (Yes / No)

---

## Phase 1 — Data Setup and SQL Exploration

- Loaded raw CSV into SQLite database
- Cleaned 11 rows with missing TotalCharges values
- Created binary ChurnFlag column (1 = churned, 0 = retained)
- Created a numeric view for ML phase
- Wrote 8 exploratory SQL queries covering churn by contract, internet service, tenure, payment method, and revenue at risk

---

## Phase 2 — Python EDA and Visualizations

Generated 8 charts using Matplotlib and Seaborn:

- Churn distribution (donut chart)
- Churn rate by contract type
- Churn rate by tenure group
- Monthly charges distribution — churned vs retained
- Churn rate by internet service
- Feature correlation heatmap
- Churn rate by payment method
- Segment risk heatmap (contract x internet service)

---

## Phase 3 — Machine Learning Model

**Feature engineering:**
- Encoded 10 binary categorical columns
- Applied one-hot encoding to contract, internet service, payment method, and tenure group
- Dropped non-predictive columns (customer ID, total charges)

**Models trained:**
- Logistic Regression
- Random Forest (selected as best model)

**Model performance (Random Forest):**

| Metric | Value |
|--------|-------|
| AUC Score | 0.8372 |
| Accuracy | 79.5% |
| Precision (churners) | 65.0% |
| Recall (churners) | 49.2% |

The model correctly identifies 184 out of 374 churners in the test set. The AUC of 0.8372 indicates strong discriminative ability between churned and retained customers.

---

## Phase 4 — Tableau Dashboard

Exported Tableau-ready CSV files and built an interactive dashboard in Tableau Public containing:

- KPI summary: total customers, churn rate, monthly revenue at risk, model AUC
- Churn by contract type (bar chart)
- Churn by tenure group (bar chart)
- Segment risk heatmap — contract type vs internet service
- Revenue at risk — churned vs retained monthly revenue
- Top 100 customers to save, ranked by expected saved monthly revenue
- Threshold business-impact curve showing the trade-off between campaign reach and net value

**Dashboard:** https://public.tableau.com/app/profile/avinash.ghai/viz/TelecomCustomerChurnAnalysis_17824221845390/Dashboard1

---

## Phase 5 — Threshold Optimization & Business Impact

Rather than using the default 0.50 classification threshold, the project evaluates thresholds from 0.05 to 0.94 using the held-out test set. Each option is assessed using precision, recall, F1-score, customers contacted, and estimated campaign value.

**Business assumptions:**

- Outreach cost: **$15 per customer**
- Retention success rate: **30%**
- Revenue estimate: expected **monthly** revenue retained

The best operating threshold was **0.67**. It focuses outreach on **129** high-confidence customers and produces an estimated **net monthly value of $420.81**. In comparison, a lower 0.30 threshold contacted 568 customers but produced an estimated loss because outreach costs outweighed expected saved revenue.

The chosen threshold is saved in `model/best_threshold.txt`; `phase_4_tableau.py` reads it to flag customers for contact and rank the retention list.

> These business values are scenario estimates based on the stated assumptions. In production, the contact cost and retention-success rate should be validated with historical campaign outcomes or an A/B test.

---

## Phase 6 — GenAI-Powered Retention Emails

Identifying a customer as high-risk is only half the job — someone still has to write the outreach message. `phase_6_genai_offers.py` closes that gap by generating a personalized retention email for each of the top 100 flagged customers, using Groq's `openai/gpt-oss-120b` model.

Each email is grounded in the customer's actual profile — contract type, tenure, monthly charge, specific churn risk reasons, and the existing rule-based recommended offer — so the model is turning a data-driven decision into natural language, not inventing one from scratch.

**Example output:**

> *"We've loved having you with us for the past three months on our fast-lane fiber optic plan, and we want to keep your connection running smoothly at the best value. To make staying even easier, we're adding a 10% discount to your current $105.90 monthly bill and offering you an upgrade to a 12-month contract that locks in this reduced rate..."*

**Design decisions:**
- Only the 100 highest-value flagged customers get an LLM call, not the full customer base — keeps runtime and API cost low while covering the segment that's actually going to be contacted
- Falls back to the original rule-based offer text if no API key is set, the call fails, or the model returns an empty response — the pipeline never breaks even if the GenAI step does
- Uses `reasoning_effort="low"` since `gpt-oss-120b` is a reasoning model that can otherwise spend its entire token budget on hidden reasoning before writing the actual email

Output saved to `tableau_data/12_genai_retention_emails.csv`.

---

## How to Run

**Prerequisites:**
```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq API key:
```
GROQ_API_KEY=your_key_here
```

**Run in order:**
```bash
python phase_1.py
python phase_2.py
python phase_3.py
python phase_5_threshold_analysis.py
python phase_4_tableau.py
python phase_6_genai_offers.py
```

`phase_5_threshold_analysis.py` must run before `phase_4_tableau.py` — it writes `model/best_threshold.txt`, which `phase_4_tableau.py` depends on to flag customers for contact. `phase_6_genai_offers.py` must run after `phase_4_tableau.py`, since it reads `tableau_data/09_top_customers_to_save.csv`.

---

## Tools and Technologies

| Category | Tool |
|----------|------|
| Database | SQLite |
| Data manipulation | Python, Pandas |
| Visualization | Matplotlib, Seaborn, Tableau Public |
| Machine learning | Scikit-learn (Logistic Regression, Random Forest) |
| GenAI | Groq API (`openai/gpt-oss-120b`) |
| IDE | VS Code |
| Version control | Git, GitHub |

---

## Business Recommendations

1. Prioritize retention campaigns for month-to-month fiber optic customers — 54.6% churn rate with average monthly charge of $87
2. Incentivize customers to switch from electronic check to automatic payment — reduces churn risk by approximately 30 percentage points
3. Focus onboarding programs on the first 12 months — nearly half of new customers churn before their first year
4. Offer contract upgrade discounts to month-to-month customers — two-year contract customers churn at only 2.8%
5. Bundle tech support with fiber optic plans — lack of support is a significant churn driver for high-paying customers

---

## Author

Avinash Ghai
