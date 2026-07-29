import sqlite3
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

os.makedirs("tableau_data", exist_ok=True)

RETENTION_SUCCESS_RATE = 0.30
CONTACT_COST = 15.0
THRESHOLDS_TO_COMPARE = [0.30, 0.40, 0.50]

print("=" * 55)
print("  PHASE 5: Threshold Optimization & Business Impact")
print("=" * 55)

conn = sqlite3.connect("churn.db")
df = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()
print(f"\n✓ Loaded {len(df):,} rows")

print("\n[1/5] Rebuilding feature set (matches phase_3.py)...")

df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49+"]
)

binary_map = {"Yes": 1, "No": 0}
binary_cols = [
    "partner", "dependents", "phoneservice", "paperlessbilling",
    "onlinesecurity", "onlinebackup", "deviceprotection",
    "techsupport", "streamingtv", "streamingmovies"
]
for col in binary_cols:
    df[col] = df[col].map(binary_map).fillna(0).astype(int)

df["gender"] = (df["gender"] == "Male").astype(int)
df["seniorcitizen"] = df["seniorcitizen"].astype(int)

df = pd.get_dummies(df, columns=["contract", "internetservice",
                                  "paymentmethod", "multiplelines",
                                  "tenure_group"], drop_first=True)

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

drop_cols = ["customerid", "churn", "totalcharges"]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

X = df.drop(columns=["churnflag"])
y = df["churnflag"]

print("\n[2/5] Reproducing the same train/test split (random_state=42)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

monthly_charge_test = X_test["monthlycharges"].reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(f"  ✓ Test set: {X_test.shape[0]:,} rows")

print("\n[3/5] Loading saved model...")

model = pickle.load(open("model/churn_model.pkl", "rb"))
feature_names = pickle.load(open("model/feature_names.pkl", "rb"))
X_test = X_test[feature_names]

if isinstance(model, LogisticRegression):
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    X_test_input = scaler.transform(X_test)
else:
    X_test_input = X_test

y_prob = model.predict_proba(X_test_input)[:, 1]
print(f"  ✓ Generated probabilities for {len(y_prob):,} test customers")

print("\n[4/5] Evaluating thresholds...")


def evaluate_thresholds(y_true, y_prob, monthly_charge, thresholds):
    results = []
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    monthly_charge = np.asarray(monthly_charge)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        n_contacted = int(y_pred.sum())

        expected_saved_revenue = (
            y_prob[y_pred == 1] * monthly_charge[y_pred == 1] * RETENTION_SUCCESS_RATE
        ).sum()
        total_contact_cost = n_contacted * CONTACT_COST
        net_value = expected_saved_revenue - total_contact_cost

        results.append({
            "threshold": t,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "customers_contacted": n_contacted,
            "expected_saved_revenue": round(expected_saved_revenue, 2),
            "total_contact_cost": round(total_contact_cost, 2),
            "net_value": round(net_value, 2),
        })

    return pd.DataFrame(results)


comparison_table = evaluate_thresholds(y_test, y_prob, monthly_charge_test, THRESHOLDS_TO_COMPARE)
print("\n" + comparison_table.to_string(index=False))

fine_grid = np.arange(0.05, 0.95, 0.01)
full_table = evaluate_thresholds(y_test, y_prob, monthly_charge_test, fine_grid)
best_row = full_table.loc[full_table["net_value"].idxmax()]
best_threshold = best_row["threshold"]

print(f"\n★ Optimal threshold by net business value: {best_threshold:.2f}")
print(f"  Net value at optimum: ${best_row['net_value']:,.2f}")
print(f"  Customers contacted: {int(best_row['customers_contacted']):,}")

print("\n[5/5] Saving outputs...")
comparison_table.to_csv("tableau_data/10_threshold_comparison.csv", index=False)
full_table.to_csv("tableau_data/11_threshold_full_curve.csv", index=False)

with open("model/best_threshold.txt", "w") as f:
    f.write(str(round(float(best_threshold), 2)))

print("  ✓ tableau_data/10_threshold_comparison.csv")
print("  ✓ tableau_data/11_threshold_full_curve.csv")
print("  ✓ model/best_threshold.txt")

print("=" * 55)
print("  PHASE 5 COMPLETE")
print("=" * 55)