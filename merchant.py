# export_excel_dashboard.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings, os
warnings.filterwarnings("ignore")

# ---------------------------
# 0. Config / paths
# ---------------------------
TRAIN_CSV = "fraudtrain.csv"
TEST_CSV  = "fraudtest.csv"
OUT_XLSX  = "merchant_risk_dashboard.xlsx"

# ---------------------------
# 1. Load data
# ---------------------------
if not (os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV)):
    raise FileNotFoundError(f"Place '{TRAIN_CSV}' and '{TEST_CSV}' in this folder and re-run script.")

train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
df = pd.concat([train, test], ignore_index=True)

# parse date, clip negative amounts
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df["amt"] = df["amt"].clip(lower=0)
df["date"] = df["trans_date_trans_time"].dt.date

# ---------------------------
# 2. Merchant aggregation/features
# ---------------------------
merchant_grp = df.groupby("merchant").agg(
    total_transactions=("merchant", "count"),
    fraud_transactions=("is_fraud", "sum"),
    avg_transaction_amt=("amt", "mean"),
    max_transaction_amt=("amt", "max"),
    std_transaction_amt=("amt", "std"),
    category_mode=("category", lambda x: x.value_counts().idxmax() if len(x)>0 else np.nan),
    dispute_rate=("category", lambda x: np.mean(x == "misc_pos"))
).reset_index()

merchant_grp["fraud_rate"] = merchant_grp["fraud_transactions"] / merchant_grp["total_transactions"]

# avg transactions per day (velocity)
merchant_daily = df.groupby(["merchant", "date"]).size().reset_index(name="daily_txn")
velocity = merchant_daily.groupby("merchant")["daily_txn"].mean().reset_index(name="avg_txn_per_day")
merchant_grp = merchant_grp.merge(velocity, on="merchant", how="left")

# high value rate (>200)
high_val = df[df["amt"] > 200].groupby("merchant")["amt"].count().rename("high_val_count")
merchant_grp = merchant_grp.merge(high_val, on="merchant", how="left")
merchant_grp["high_val_count"] = merchant_grp["high_val_count"].fillna(0)
merchant_grp["high_value_rate"] = merchant_grp["high_val_count"] / merchant_grp["total_transactions"].replace(0,1)

# category risk: fraud frequency by category
category_risk = df.groupby("category")["is_fraud"].mean().to_dict()
merchant_grp["category_risk"] = merchant_grp["category_mode"].map(category_risk).fillna(0)

# handle NaNs
merchant_grp["std_transaction_amt"] = merchant_grp["std_transaction_amt"].fillna(0)

# ---------------------------
# 3. MRI 2.0 scorecard (normalized)
# ---------------------------
merchant_grp["avg_transaction_amt_scaled"] = merchant_grp["avg_transaction_amt"] / (merchant_grp["avg_transaction_amt"].max() or 1)
merchant_grp["std_transaction_amt_scaled"] = merchant_grp["std_transaction_amt"] / (merchant_grp["std_transaction_amt"].max() or 1)

merchant_grp["MRI_raw"] = (
    0.45 * merchant_grp["fraud_rate"] +
    0.15 * merchant_grp["dispute_rate"] +
    0.10 * merchant_grp["avg_transaction_amt_scaled"] +
    0.10 * merchant_grp["std_transaction_amt_scaled"] +
    0.10 * merchant_grp["high_value_rate"] +
    0.10 * merchant_grp["category_risk"]
)

# normalize to 0-1
minr, maxr = merchant_grp["MRI_raw"].min(), merchant_grp["MRI_raw"].max()
merchant_grp["MRI_normalized"] = (merchant_grp["MRI_raw"] - minr) / (maxr - minr) if maxr != minr else 0.0
merchant_grp["risk_bucket"] = pd.cut(merchant_grp["MRI_normalized"], bins=[-1, .33, .66, 1.0], labels=["Low", "Medium", "High"])

# ---------------------------
# 4. Clustering
# ---------------------------
features = merchant_grp[["fraud_rate","avg_transaction_amt","std_transaction_amt","high_value_rate","avg_txn_per_day","category_risk","MRI_normalized"]]
imputer = SimpleImputer(strategy="median")
scaled = StandardScaler().fit_transform(imputer.fit_transform(features))
kmeans = KMeans(n_clusters=4, random_state=42)
merchant_grp["cluster"] = kmeans.fit_predict(scaled).astype(int)

# label clusters by MRI mean
cluster_risk_mean = merchant_grp.groupby("cluster")["MRI_normalized"].mean().sort_values()
cluster_order = list(cluster_risk_mean.index)
label_map = {cluster_order[i]: lab for i, lab in enumerate(["Low","Medium","High","Critical"][:len(cluster_order)])}
merchant_grp["cluster_risk_label"] = merchant_grp["cluster"].map(label_map)

# ---------------------------
# 5. Expected loss + XGBoost forecasting
# ---------------------------
merchant_grp["expected_loss"] = merchant_grp["fraud_rate"] * merchant_grp["avg_transaction_amt"] * merchant_grp["total_transactions"]
X = features.fillna(0)
y = merchant_grp["expected_loss"].fillna(0)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.8)
model.fit(Xtr, ytr)
preds = model.predict(Xte)
mse = mean_squared_error(yte, preds)

merchant_grp["predicted_loss"] = model.predict(X.fillna(0))

# ---------------------------
# 6. Monthly anomaly detection for top MRI merchant (z-score)
# ---------------------------
top_merchant = merchant_grp.sort_values("MRI_normalized", ascending=False).iloc[0]["merchant"]
monthly = df[df["merchant"] == top_merchant].set_index("trans_date_trans_time").resample("M")["is_fraud"].sum().reset_index()
monthly.columns = ["ds","y"]
if len(monthly) >= 3:
    monthly["rolling_mean"] = monthly["y"].rolling(3, min_periods=1).mean()
    monthly["rolling_std"]  = monthly["y"].rolling(3, min_periods=1).std().fillna(0)
    monthly["z_score"] = (monthly["y"] - monthly["rolling_mean"]) / (monthly["rolling_std"].replace(0, np.nan))
    monthly["anomaly"] = monthly["z_score"] > 2
else:
    monthly["rolling_mean"] = np.nan
    monthly["rolling_std"] = np.nan
    monthly["z_score"] = np.nan
    monthly["anomaly"] = False

# ---------------------------
# 7. Human-readable risk reason
# ---------------------------
merchant_grp["risk_reason"] = ""
merchant_grp["risk_reason"] += np.where(merchant_grp["fraud_rate"] > merchant_grp["fraud_rate"].median(), "High fraud rate | ", "")
merchant_grp["risk_reason"] += np.where(merchant_grp["dispute_rate"] > merchant_grp["dispute_rate"].median(), "High dispute rate | ", "")
merchant_grp["risk_reason"] += np.where(merchant_grp["high_value_rate"] > merchant_grp["high_value_rate"].median(), "High-value transactions | ", "")
merchant_grp["risk_reason"] += np.where(merchant_grp["category_risk"] > merchant_grp["category_risk"].median(), "Risky business category | ", "")
merchant_grp["risk_reason"] = merchant_grp["risk_reason"].str.rstrip(" | ")

# ---------------------------
# 8. Summaries & Top 20
# ---------------------------
top20 = merchant_grp.sort_values("MRI_normalized", ascending=False).head(20).reset_index(drop=True)
cluster_summary = merchant_grp.groupby("cluster_risk_label").agg(
    num_merchants=("merchant","count"),
    avg_MRI=("MRI_normalized","mean"),
    total_predicted_loss=("predicted_loss","sum")
).reset_index()

# ---------------------------
# 9. Export to Excel with charts
# ---------------------------
out_path = OUT_XLSX
with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    merchant_grp.to_excel(writer, sheet_name="merchant_scores", index=False)
    top20.to_excel(writer, sheet_name="top20", index=False)
    cluster_summary.to_excel(writer, sheet_name="cluster_summary", index=False)
    monthly.to_excel(writer, sheet_name="monthly_top_merchant", index=False)

    workbook  = writer.book
    ws_top20  = writer.sheets["top20"]
    ws_scores = writer.sheets["merchant_scores"]

    # Bar chart: Top20 MRI
    # find column index for MRI_normalized in top20
    top20_cols = list(top20.columns)
    cat_col_idx = top20_cols.index("merchant")  # 0-based inside the excel sheet
    val_col_idx = top20_cols.index("MRI_normalized")
    chart1 = workbook.add_chart({'type':'column'})
    chart1.add_series({
        'name': 'MRI_normalized',
        'categories': ['top20', 1, cat_col_idx, len(top20), cat_col_idx],
        'values':     ['top20', 1, val_col_idx, len(top20), val_col_idx],
    })
    chart1.set_title({'name':'Top 20 Merchants by MRI'})
    chart1.set_x_axis({'name':'Merchant', 'interval_unit':1})
    chart1.set_y_axis({'name':'MRI_normalized'})
    ws_top20.insert_chart('K2', chart1, {'x_scale':1.3, 'y_scale':1.3})

    # Scatter: expected vs predicted loss
    ms_cols = list(merchant_grp.columns)
    exp_idx = ms_cols.index("expected_loss")
    pred_idx = ms_cols.index("predicted_loss")
    chart2 = workbook.add_chart({'type':'scatter', 'subtype':'straight_with_markers'})
    chart2.add_series({
        'name':'Expected vs Predicted Loss',
        'categories':['merchant_scores', 1, exp_idx, len(merchant_grp), exp_idx],
        'values':['merchant_scores', 1, pred_idx, len(merchant_grp), pred_idx],
    })
    chart2.set_title({'name':'Expected vs Predicted Loss (merchant level)'})
    chart2.set_x_axis({'name':'Expected Loss'})
    chart2.set_y_axis({'name':'Predicted Loss'})
    ws_scores.insert_chart('L2', chart2, {'x_scale':1.3, 'y_scale':1.3})

# ---------------------------
# 10. Done: print summary
# ---------------------------
print("Excel dashboard saved to:", out_path)
print(f"Top merchant (for monthly anomalies): {top_merchant}")
print("Top 5 merchants (preview):")
print(top20[["merchant","fraud_rate","MRI_normalized","cluster_risk_label","predicted_loss"]].head())
