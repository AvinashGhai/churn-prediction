import sqlite3
import pandas as pd

CSV_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DB_FILE = "churn.db"

print("=" * 55)
print("  PHASE 1: Data Setup & SQL Foundation")
print("=" * 55)

print("\n[1/4] Loading CSV...")
df = pd.read_csv(CSV_FILE)
print(f"  ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")

print("\n[2/4] Cleaning data...")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

before = len(df)
df.dropna(subset=["TotalCharges"], inplace=True)
print(f"  ✓ Removed {before - len(df)} rows with missing TotalCharges")

df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

print(f"  ✓ ChurnFlag created  |  Overall churn rate: {df['churnflag'].mean():.1%}")
print(f"  ✓ Final shape: {df.shape}")

print(f"\n[3/4] Loading into SQLite → {DB_FILE}...")

conn = sqlite3.connect(DB_FILE)

df.to_sql("customers", conn, if_exists="replace", index=False)

conn.execute("DROP VIEW IF EXISTS customers_numeric")

conn.execute("""
    CREATE VIEW customers_numeric AS
    SELECT
        customerid,
        tenure,
        monthlycharges,
        totalcharges,
        churnflag,
        CASE WHEN gender = 'Male' THEN 1 ELSE 0 END AS gender_male,
        CASE WHEN seniorcitizen = 1 THEN 1 ELSE 0 END AS is_senior,
        CASE WHEN partner = 'Yes' THEN 1 ELSE 0 END AS has_partner,
        CASE WHEN dependents = 'Yes' THEN 1 ELSE 0 END AS has_dependents,
        CASE WHEN phoneservice = 'Yes' THEN 1 ELSE 0 END AS has_phone,
        CASE WHEN multiplelines = 'Yes' THEN 1 ELSE 0 END AS has_multiplelines,
        CASE WHEN internetservice = 'Fiber optic' THEN 1 ELSE 0 END AS fiber_optic,
        CASE WHEN internetservice = 'DSL' THEN 1 ELSE 0 END AS has_dsl,
        CASE WHEN onlinesecurity = 'Yes' THEN 1 ELSE 0 END AS has_security,
        CASE WHEN onlinebackup = 'Yes' THEN 1 ELSE 0 END AS has_backup,
        CASE WHEN techsupport = 'Yes' THEN 1 ELSE 0 END AS has_techsupport,
        CASE WHEN contract = 'Month-to-month' THEN 1 ELSE 0 END AS monthly_contract,
        CASE WHEN contract = 'Two year' THEN 1 ELSE 0 END AS twoyear_contract,
        CASE WHEN paperlessbilling = 'Yes' THEN 1 ELSE 0 END AS paperless_billing,
        CASE WHEN paymentmethod LIKE '%Electronic%' THEN 1 ELSE 0 END AS electronic_payment
    FROM customers
""")

conn.commit()

print(f"  ✓ Table 'customers' created with {len(df):,} rows")
print("  ✓ View 'customers_numeric' created for ML phase")

print("\n[4/4] Running exploratory SQL queries...\n")

queries = {
    "── Q1: Overall churn count ──": """
        SELECT
            churn,
            COUNT(*) AS total_customers,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 1) AS pct
        FROM customers
        GROUP BY churn
    """,

    "── Q2: Churn rate by contract type ──": """
        SELECT
            contract,
            COUNT(*) AS customers,
            SUM(churnflag) AS churned,
            ROUND(AVG(churnflag) * 100, 1) AS churn_rate_pct,
            ROUND(AVG(monthlycharges), 2) AS avg_monthly_charge
        FROM customers
        GROUP BY contract
        ORDER BY churn_rate_pct DESC
    """,

    "── Q3: Churn rate by internet service ──": """
        SELECT
            internetservice,
            COUNT(*) AS customers,
            SUM(churnflag) AS churned,
            ROUND(AVG(churnflag) * 100, 1) AS churn_rate_pct
        FROM customers
        GROUP BY internetservice
        ORDER BY churn_rate_pct DESC
    """,

    "── Q4: Tenure buckets — who churns earliest? ──": """
        SELECT
            CASE
                WHEN tenure <= 12 THEN '0-12 months'
                WHEN tenure <= 24 THEN '13-24 months'
                WHEN tenure <= 48 THEN '25-48 months'
                ELSE '49+ months'
            END AS tenure_group,
            COUNT(*) AS customers,
            SUM(churnflag) AS churned,
            ROUND(AVG(churnflag) * 100, 1) AS churn_rate_pct,
            ROUND(AVG(monthlycharges), 2) AS avg_monthly_charges
        FROM customers
        GROUP BY tenure_group
        ORDER BY churn_rate_pct DESC
    """,

    "── Q5: Payment method vs churn ──": """
        SELECT
            paymentmethod,
            COUNT(*) AS customers,
            SUM(churnflag) AS churned,
            ROUND(AVG(churnflag) * 100, 1) AS churn_rate_pct
        FROM customers
        GROUP BY paymentmethod
        ORDER BY churn_rate_pct DESC
    """,

    "── Q6: High-risk segment (fiber + month-to-month) ──": """
        SELECT
            COUNT(*) AS total_segment,
            SUM(churnflag) AS churned,
            ROUND(AVG(churnflag) * 100, 1) AS churn_rate_pct,
            ROUND(AVG(monthlycharges), 2) AS avg_monthly_charges,
            ROUND(AVG(tenure), 1) AS avg_tenure_months
        FROM customers
        WHERE internetservice = 'Fiber optic'
          AND contract = 'Month-to-month'
    """,

    "── Q7: Revenue at risk (churned customers) ──": """
        SELECT
            churn,
            COUNT(*) AS customers,
            ROUND(SUM(monthlycharges), 2) AS total_monthly_revenue,
            ROUND(AVG(monthlycharges), 2) AS avg_monthly_charge,
            ROUND(SUM(totalcharges), 2) AS total_lifetime_revenue
        FROM customers
        GROUP BY churn
    """,

    "── Q8: Contract + Internet Service Churn ──": """
        SELECT
            contract,
            internetservice,
            COUNT(*) AS customers,
            ROUND(AVG(churnflag) * 100, 1) AS churn_rate_pct
        FROM customers
        GROUP BY contract, internetservice
        ORDER BY churn_rate_pct DESC
    """
}

for title, sql in queries.items():
    print(f"\n{title}")
    result = pd.read_sql_query(sql, conn)
    print(result.to_string(index=False))

conn.close()

print("\n" + "=" * 55)
print("  Phase 1 Complete!")
print(f"  SQLite Database Created: {DB_FILE}")
print("=" * 55)

