import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)

os.makedirs("charts", exist_ok=True)
os.makedirs("model",  exist_ok=True)

BG_COLOR   = "#f8f9fa"
MAIN_COLOR = "#e74c3c"
SAFE_COLOR = "#2ecc71"
BLUE_COLOR = "#3498db"

plt.rcParams.update({
    "figure.facecolor" : BG_COLOR,
    "axes.facecolor"   : BG_COLOR,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "sans-serif",
    "axes.titlesize"   : 14,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 11,
})

print("=" * 55)
print("  PHASE 3: Feature Engineering & ML Model")
print("=" * 55)


conn  = sqlite3.connect("churn.db")
df    = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()
print(f"\n✓ Loaded {len(df):,} rows")


print("\n[1/6] Feature engineering...")

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

df["gender"]        = (df["gender"] == "Male").astype(int)
df["seniorcitizen"] = df["seniorcitizen"].astype(int)

df = pd.get_dummies(df, columns=["contract", "internetservice",
                                  "paymentmethod", "multiplelines",
                                  "tenure_group"], drop_first=True)

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

drop_cols = ["customerid", "churn", "totalcharges"]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

print(f"  ✓ Features after engineering: {df.shape[1] - 1}")


print("\n[2/6] Splitting data...")

X = df.drop(columns=["churnflag"])
y = df["churnflag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  ✓ Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
print(f"  ✓ Churn rate in test set: {y_test.mean():.1%}")


print("\n[3/6] Training models...")

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
lr_pred  = lr.predict(X_test_sc)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_proba)
lr_acc   = accuracy_score(y_test, lr_pred)
print(f"  ✓ Logistic Regression  →  AUC: {lr_auc:.4f}  |  Accuracy: {lr_acc:.4f}")

rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_proba)
rf_acc   = accuracy_score(y_test, rf_pred)
print(f"  ✓ Random Forest        →  AUC: {rf_auc:.4f}  |  Accuracy: {rf_acc:.4f}")

best_model     = rf if rf_auc >= lr_auc else lr
best_name      = "Random Forest" if rf_auc >= lr_auc else "Logistic Regression"
best_proba     = rf_proba if rf_auc >= lr_auc else lr_proba
best_pred      = rf_pred if rf_auc >= lr_auc else lr_pred
print(f"\n  ★ Best model: {best_name} (AUC {max(rf_auc, lr_auc):.4f})")


print("\n[4/6] Saving model & scaler...")
pickle.dump(best_model, open("model/churn_model.pkl", "wb"))
pickle.dump(scaler,     open("model/scaler.pkl",      "wb"))
pickle.dump(list(X.columns), open("model/feature_names.pkl", "wb"))
print("  ✓ model/churn_model.pkl")
print("  ✓ model/scaler.pkl")
print("  ✓ model/feature_names.pkl")


print("\n[5/6] Generating evaluation charts...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
ax = axes[0]
ax.plot(fpr_lr, tpr_lr, color=BLUE_COLOR,  lw=2, label=f"Logistic Regression (AUC = {lr_auc:.3f})")
ax.plot(fpr_rf, tpr_rf, color=MAIN_COLOR,  lw=2, label=f"Random Forest       (AUC = {rf_auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Model Comparison")
ax.legend(fontsize=10)

cm = confusion_matrix(y_test, best_pred)
ax = axes[1]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Predicted: Stay", "Predicted: Churn"],
            yticklabels=["Actual: Stay",    "Actual: Churn"],
            annot_kws={"size": 14, "weight": "bold"},
            cbar=False, linewidths=0.5)
ax.set_title(f"Confusion Matrix — {best_name}")
ax.set_xlabel("")
ax.set_ylabel("")

plt.tight_layout()
plt.savefig("charts/09_model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved charts/09_model_evaluation.png")

feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
top15    = feat_imp.sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 7))
colors  = [MAIN_COLOR if v > top15.mean() else BLUE_COLOR for v in top15.values]
bars    = ax.barh(top15.index, top15.values, color=colors, height=0.6)
ax.set_xlabel("Feature Importance Score")
ax.set_title(f"Top 15 Features — {best_name}\n(red = above average importance)")
for bar, val in zip(bars, top15.values):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("charts/10_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved charts/10_feature_importance.png")


print("\n[6/6] Classification report...")
print(f"\n{classification_report(y_test, best_pred, target_names=['Retained', 'Churned'])}")

tn, fp, fn, tp = cm.ravel()
precision  = tp / (tp + fp)
recall     = tp / (tp + fn)
caught_rev = df.shape[0]

print("=" * 55)
print("  PHASE 3 COMPLETE")
print("=" * 55)
print(f"""
Best Model      : {best_name}
AUC Score       : {max(rf_auc, lr_auc):.4f}
Accuracy        : {max(rf_acc, lr_acc):.4f}
Precision       : {precision:.4f}  (of predicted churners, {precision:.1%} are correct)
Recall          : {recall:.4f}  (caught {recall:.1%} of actual churners)

Confusion Matrix breakdown
  True Positives  (correctly caught churners) : {tp}
  False Positives (wrongly flagged as churn)  : {fp}
  True Negatives  (correctly retained)        : {tn}
  False Negatives (missed churners)           : {fn}

Files saved
  model/churn_model.pkl
  model/scaler.pkl
  model/feature_names.pkl
  charts/09_model_evaluation.png
  charts/10_feature_importance.png

Next → Run phase4_streamlit.py to build the dashboard
""")