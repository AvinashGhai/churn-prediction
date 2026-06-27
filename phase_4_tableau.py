import sqlite3
import pandas as pd
import numpy as np
import pickle
import os

os.makedirs("tableau_data", exist_ok=True)

conn = sqlite3.connect("churn.db")
df = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()

df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"]
)

df["churn_label"] = df["churn"].map({"Yes": "Churned", "No": "Retained"})

df["revenue_segment"] = pd.cut(
    df["monthlycharges"],
    bins=[0, 35, 65, 95, 200],
    labels=["Low ($0-35)", "Medium ($35-65)", "High ($65-95)", "Premium ($95+)"]
)

df["risk_score"] = (
    (df["contract"] == "Month-to-month").astype(int) * 3 +
    (df["internetservice"] == "Fiber optic").astype(int) * 2 +
    (df["paymentmethod"] == "Electronic check").astype(int) * 2 +
    (df["tenure"] <= 12).astype(int) * 2 +
    (df["techsupport"] == "No").astype(int) * 1 +
    (df["onlinesecurity"] == "No").astype(int) * 1
)

df["risk_level"] = pd.cut(
    df["risk_score"],
    bins=[-1, 2, 5, 11],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

df.to_csv("tableau_data/01_customers_main.csv", index=False)

contract_summary = (df.groupby("contract")
    .agg(
        total_customers=("churnflag", "count"),
        churned=("churnflag", "sum"),
        churn_rate_pct=("churnflag", lambda x: round(x.mean() * 100, 1)),
        avg_monthly=("monthlycharges", lambda x: round(x.mean(), 2)),
        total_revenue=("monthlycharges", "sum")
    ).reset_index())

internet_summary = (df.groupby("internetservice")
    .agg(
        total_customers=("churnflag", "count"),
        churned=("churnflag", "sum"),
        churn_rate_pct=("churnflag", lambda x: round(x.mean() * 100, 1)),
        avg_monthly=("monthlycharges", lambda x: round(x.mean(), 2)),
        total_revenue=("monthlycharges", "sum")
    ).reset_index().rename(columns={"internetservice": "segment"}))

payment_summary = (df.groupby("paymentmethod")
    .agg(
        total_customers=("churnflag", "count"),
        churned=("churnflag", "sum"),
        churn_rate_pct=("churnflag", lambda x: round(x.mean() * 100, 1)),
        avg_monthly=("monthlycharges", lambda x: round(x.mean(), 2)),
    ).reset_index().rename(columns={"paymentmethod": "segment"}))

tenure_summary = (df.groupby("tenure_group", observed=True)
    .agg(
        total_customers=("churnflag", "count"),
        churned=("churnflag", "sum"),
        churn_rate_pct=("churnflag", lambda x: round(x.mean() * 100, 1)),
        avg_monthly=("monthlycharges", lambda x: round(x.mean(), 2)),
    ).reset_index().rename(columns={"tenure_group": "segment"}))

contract_summary.to_csv("tableau_data/02_contract_summary.csv", index=False)
internet_summary.to_csv("tableau_data/03_internet_summary.csv", index=False)
payment_summary.to_csv("tableau_data/04_payment_summary.csv", index=False)
tenure_summary.to_csv("tableau_data/05_tenure_summary.csv", index=False)

heatmap = (df.groupby(["contract", "internetservice"])
    .agg(
        customers=("churnflag", "count"),
        churned=("churnflag", "sum"),
        churn_rate_pct=("churnflag", lambda x: round(x.mean() * 100, 1)),
        avg_monthly=("monthlycharges", lambda x: round(x.mean(), 2)),
        monthly_revenue_at_risk=("monthlycharges", lambda x: round(
            x[df.loc[x.index, "churn"] == "Yes"].sum(), 2))
    ).reset_index())

heatmap.to_csv("tableau_data/06_segment_heatmap.csv", index=False)

model = pickle.load(open("model/churn_model.pkl", "rb"))
features = pickle.load(open("model/feature_names.pkl", "rb"))

df_ml = pd.read_sql_query("SELECT * FROM customers", sqlite3.connect("churn.db"))

binary_cols = ["partner", "dependents", "phoneservice", "paperlessbilling",
               "onlinesecurity", "onlinebackup", "deviceprotection",
               "techsupport", "streamingtv", "streamingmovies"]
binary_map = {"Yes": 1, "No": 0}
for col in binary_cols:
    df_ml[col] = df_ml[col].map(binary_map).fillna(0).astype(int)

df_ml["gender"] = (df_ml["gender"] == "Male").astype(int)
df_ml["seniorcitizen"] = df_ml["seniorcitizen"].astype(int)
df_ml["tenure_group"] = pd.cut(df_ml["tenure"], bins=[0, 12, 24, 48, 72],
                                labels=["0-12", "13-24", "25-48", "49+"])

df_ml = pd.get_dummies(df_ml, columns=["contract", "internetservice",
                                        "paymentmethod", "multiplelines",
                                        "tenure_group"], drop_first=True)

bool_cols = df_ml.select_dtypes(include="bool").columns
df_ml[bool_cols] = df_ml[bool_cols].astype(int)

df_ml.drop(columns=["churn", "totalcharges"], inplace=True, errors="ignore")

X = df_ml.drop(columns=["churnflag", "customerid"], errors="ignore")
for col in features:
    if col not in X.columns:
        X[col] = 0
X = X[features]

churn_prob = model.predict_proba(X)[:, 1]

predictions = pd.read_sql_query("SELECT * FROM customers", sqlite3.connect("churn.db"))
predictions["tenure_group"] = pd.cut(predictions["tenure"], bins=[0, 12, 24, 48, 72],
                                      labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"])
predictions["churn_probability"] = (churn_prob * 100).round(1)
predictions["predicted_churn"] = (churn_prob >= 0.5).astype(int)
predictions["risk_level"] = pd.cut(churn_prob,
                                    bins=[-0.01, 0.35, 0.6, 1.01],
                                    labels=["Low Risk", "Medium Risk", "High Risk"])

predictions["retention_priority_score"] = (
    predictions["churn_probability"] * predictions["monthlycharges"]
).round(2)

def get_offer(row):
    if row["contract"] == "Month-to-month" and row["internetservice"] == "Fiber optic":
        return "10% discount + contract upgrade offer"
    elif row["contract"] == "Month-to-month" and row["tenure"] <= 12:
        return "Free onboarding support + loyalty discount"
    elif row["paymentmethod"] == "Electronic check":
        return "Switch to auto-pay + $5/month discount"
    elif row["techsupport"] == "No":
        return "Free tech support trial for 3 months"
    elif row["internetservice"] == "DSL" and row["contract"] == "Month-to-month":
        return "Upgrade to Fiber + 15% first year discount"
    else:
        return "Loyalty reward + personalized outreach"

def get_risk_reasons(row):
    reasons = []
    if row["contract"] == "Month-to-month":
        reasons.append("Month-to-month contract")
    if row["internetservice"] == "Fiber optic":
        reasons.append("Fiber optic plan")
    if row["tenure"] <= 12:
        reasons.append("New customer")
    if row["paymentmethod"] == "Electronic check":
        reasons.append("Electronic check payment")
    if row["techsupport"] == "No":
        reasons.append("No tech support")
    if row["onlinesecurity"] == "No":
        reasons.append("No online security")
    return ", ".join(reasons) if reasons else "Low risk profile"

predictions["recommended_offer"] = predictions.apply(get_offer, axis=1)
predictions["risk_reasons"] = predictions.apply(get_risk_reasons, axis=1)

predictions.to_csv("tableau_data/07_predictions.csv", index=False)

top_customers = predictions[predictions["risk_level"] == "High Risk"].sort_values(
    "retention_priority_score", ascending=False
).head(100)[[
    "customerid", "churn_probability", "monthlycharges",
    "retention_priority_score", "risk_reasons", "recommended_offer",
    "contract", "internetservice", "tenure"
]]
top_customers.to_csv("tableau_data/09_top_customers_to_save.csv", index=False)

rev = pd.DataFrame({
    "segment": ["Retained", "Churned"],
    "customers": [
        len(df[df["churn"] == "No"]),
        len(df[df["churn"] == "Yes"])
    ],
    "total_monthly_revenue": [
        round(df[df["churn"] == "No"]["monthlycharges"].sum(), 2),
        round(df[df["churn"] == "Yes"]["monthlycharges"].sum(), 2)
    ],
    "avg_monthly_charge": [
        round(df[df["churn"] == "No"]["monthlycharges"].mean(), 2),
        round(df[df["churn"] == "Yes"]["monthlycharges"].mean(), 2)
    ],
    "avg_tenure_months": [
        round(df[df["churn"] == "No"]["tenure"].mean(), 1),
        round(df[df["churn"] == "Yes"]["tenure"].mean(), 1)
    ],
})
rev.to_csv("tableau_data/08_revenue_summary.csv", index=False)

print("Export complete.")