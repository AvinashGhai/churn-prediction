import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("charts", exist_ok=True)

CHURN_COLORS = ["#2ecc71", "#e74c3c"]
MAIN_COLOR   = "#e74c3c"
SAFE_COLOR   = "#2ecc71"
BG_COLOR     = "#f8f9fa"

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
print("  PHASE 2: EDA & Visualizations")
print("=" * 55)

conn = sqlite3.connect("churn.db")
df   = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()

print(f"\n✓ Loaded {len(df):,} rows from churn.db")
print(f"  Churn rate: {df['churnflag'].mean():.1%}\n")


print("[1/8] Churn distribution...")
fig, ax = plt.subplots(figsize=(7, 5))
counts = df["churn"].value_counts()
wedges, texts, autotexts = ax.pie(
    counts,
    labels=["Retained", "Churned"],
    autopct="%1.1f%%",
    colors=CHURN_COLORS,
    startangle=90,
    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=3),
    textprops=dict(fontsize=12),
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight("bold")
    at.set_color("white")
ax.set_title("Overall Customer Churn Distribution", pad=20)
centre = plt.Circle((0, 0), 0.35, color=BG_COLOR)
ax.add_patch(centre)
ax.text(0, 0, f"{counts['Yes']:,}\nchurned", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#e74c3c")
plt.tight_layout()
plt.savefig("charts/01_churn_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved charts/01_churn_distribution.png")


print("[2/8] Churn by contract type...")
contract_churn = (df.groupby("contract")["churnflag"]
                    .mean().mul(100).reset_index()
                    .rename(columns={"churnflag": "churn_rate"})
                    .sort_values("churn_rate", ascending=False))
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(contract_churn["contract"], contract_churn["churn_rate"],
               color=[MAIN_COLOR, "#e67e22", SAFE_COLOR], height=0.5)
for bar, val in zip(bars, contract_churn["churn_rate"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontweight="bold", fontsize=12)
ax.set_xlim(0, 55)
ax.set_xlabel("Churn Rate (%)")
ax.set_title("Churn Rate by Contract Type\n(Month-to-month is 15x riskier than two-year)")
ax.axvline(df["churnflag"].mean() * 100, color="gray",
           linestyle="--", linewidth=1.2, label=f"Avg ({df['churnflag'].mean():.1%})")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("charts/02_churn_by_contract.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved charts/02_churn_by_contract.png")


print("[3/8] Churn by tenure...")
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0–12 months", "13–24 months", "25–48 months", "49+ months"]
)
tenure_churn = (df.groupby("tenure_group", observed=True)["churnflag"]
                  .mean().mul(100).reset_index()
                  .rename(columns={"churnflag": "churn_rate"}))
fig, ax = plt.subplots(figsize=(8, 5))
colors = [MAIN_COLOR, "#e67e22", "#f39c12", SAFE_COLOR]
bars = ax.bar(tenure_churn["tenure_group"], tenure_churn["churn_rate"],
              color=colors, width=0.55, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, tenure_churn["churn_rate"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{val:.1f}%", ha="center", fontweight="bold", fontsize=12)
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Customer Tenure")
ax.set_title("Churn Rate by Tenure Group\n(New customers churn almost 5x more than long-term ones)")
ax.set_ylim(0, 60)
plt.tight_layout()
plt.savefig("charts/03_churn_by_tenure.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved charts/03_churn_by_tenure.png")


print("[4/8] Monthly charges distribution...")
fig, ax = plt.subplots(figsize=(9, 5))
for label, color in [("No", SAFE_COLOR), ("Yes", MAIN_COLOR)]:
    subset = df[df["churn"] == label]["monthlycharges"]
    ax.hist(subset, bins=40, alpha=0.6, color=color,
            label=f"{'Churned' if label == 'Yes' else 'Retained'}",
            edgecolor="white", linewidth=0.5)
ax.axvline(df[df["churn"] == "Yes"]["monthlycharges"].mean(),
           color=MAIN_COLOR, linestyle="--", linewidth=2,
           label=f"Churned avg: ${df[df['churn'] == 'Yes']['monthlycharges'].mean():.0f}")
ax.axvline(df[df["churn"] == "No"]["monthlycharges"].mean(),
           color=SAFE_COLOR, linestyle="--", linewidth=2,
           label=f"Retained avg: ${df[df['churn'] == 'No']['monthlycharges'].mean():.0f}")
ax.set_xlabel("Monthly Charges ($)")
ax.set_ylabel("Number of Customers")
ax.set_title("Monthly Charges: Churned vs Retained\n(Churned customers pay $18/month more on average)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("charts/04_monthly_charges_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved charts/04_monthly_charges_dist.png")


print("[5/8] Churn by internet service...")
internet_churn = (df.groupby("internetservice")["churnflag"]
                    .mean().mul(100).reset_index()
                    .rename(columns={"churnflag": "churn_rate"})
                    .sort_values("churn_rate", ascending=False))
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(internet_churn["internetservice"], internet_churn["churn_rate"],
              color=[MAIN_COLOR, "#e67e22", SAFE_COLOR],
              width=0.5, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, internet_churn["churn_rate"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontweight="bold", fontsize=13)
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Internet Service Type")
ax.set_title("Churn Rate by Internet Service\n(Fiber optic customers churn most despite paying more)")
ax.set_ylim(0, 55)
plt.tight_layout()
plt.savefig("charts/05_churn_by_internet.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved charts/05_churn_by_internet.png")


print("[6/8] Correlation heatmap...")
conn = sqlite3.connect("churn.db")
df_num = pd.read_sql_query("SELECT * FROM customers_numeric", conn)
conn.close()
df_num = df_num.drop(columns=["customerid"], errors="ignore")
corr = df_num.corr(numeric_only=True)[["churnflag"]].drop("churnflag").sort_values("churnflag", ascending=False)
fig, ax = plt.subplots(figsize=(7, 9))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn_r",
    center=0,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"shrink": 0.6},
    annot_kws={"size": 11},
)
ax.set_title("Feature Correlation with Churn\n(red = higher churn risk, green = protective)", pad=15)
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("charts/06_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved charts/06_correlation_heatmap.png")


print("[7/8] Churn by payment method...")
pay_churn = (df.groupby("paymentmethod")["churnflag"]
               .mean().mul(100).reset_index()
               .rename(columns={"churnflag": "churn_rate"})
               .sort_values("churn_rate", ascending=True))
pay_churn["paymentmethod"] = pay_churn["paymentmethod"].str.replace(
    " (automatic)", "\n(auto)", regex=False)
fig, ax = plt.subplots(figsize=(9, 5))
colors = [SAFE_COLOR, SAFE_COLOR, "#e67e22", MAIN_COLOR]
bars = ax.barh(pay_churn["paymentmethod"], pay_churn["churn_rate"],
               color=colors, height=0.5)
for bar, val in zip(bars, pay_churn["churn_rate"]):
    ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontweight="bold", fontsize=12)
ax.set_xlim(0, 55)
ax.set_xlabel("Churn Rate (%)")
ax.set_title("Churn Rate by Payment Method\n(Electronic check users churn 3x more than credit card)")
plt.tight_layout()
plt.savefig("charts/07_churn_by_payment.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved charts/07_churn_by_payment.png")


print("[8/8] High-risk segment heatmap...")
pivot = (df.groupby(["contract", "internetservice"])["churnflag"]
           .mean().mul(100).unstack().round(1))
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".1f",
    cmap="Reds",
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 13, "weight": "bold"},
    cbar_kws={"label": "Churn Rate (%)"},
)
ax.set_title("Churn Rate (%) — Contract Type × Internet Service\n(54.6% for Fiber + Month-to-month = highest risk segment)", pad=15)
ax.set_xlabel("Internet Service")
ax.set_ylabel("Contract Type")
plt.tight_layout()
plt.savefig("charts/08_segment_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved charts/08_segment_heatmap.png")


churned  = df[df["churn"] == "Yes"]
retained = df[df["churn"] == "No"]

print("\n" + "=" * 55)
print("  PHASE 2 COMPLETE — KEY INSIGHTS")
print("=" * 55)
print(f"""
Total customers  : {len(df):,}
Churned          : {len(churned):,} ({len(churned)/len(df):.1%})
Retained         : {len(retained):,} ({len(retained)/len(df):.1%})

Avg monthly charge (churned)  : ${churned['monthlycharges'].mean():.2f}
Avg monthly charge (retained) : ${retained['monthlycharges'].mean():.2f}
Monthly revenue at risk       : ${churned['monthlycharges'].sum():,.0f}

Top Risk Factors
  1. Month-to-month contract  → 42.7% churn rate
  2. Fiber optic internet     → 41.9% churn rate
  3. Electronic check payment → 45.3% churn rate
  4. Tenure < 12 months       → 47.7% churn rate

Highest Risk Segment
  Fiber optic + Month-to-month → 54.6% churn rate
""")