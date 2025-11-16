📘 Merchant Performance & Risk Intelligence System
Using fraudtrain.csv & fraudtest.csv (Sparkov Fraud Dataset)
End-to-end Data Analytics + Machine Learning + Excel Dashboard Project
📌 1. Project Overview

This project builds a Merchant Performance & Risk Intelligence System (MRI 2.0) using real credit-card transaction data.
The system evaluates merchants based on:

Fraud patterns

Chargeback & dispute behavior

Transaction velocity

High-value transaction risk

Business category risk

Expected loss forecasting

Cluster-based merchant segmentation

The output is a fully automated Excel dashboard with charts, tables, and risk insights.

🎯 2. Features
🧠 Risk Scoring – MRI 2.0

A weighted score based on:

Fraud Rate

Dispute Rate

Avg Transaction Amount

Std Deviation of Amount

High-Value Transaction Rate

Category Fraudiness

🔍 Merchant Segmentation

Uses K-Means clustering to group merchants into:

Low Risk

Medium Risk

High Risk

Critical Risk

📉 Machine Learning

Uses XGBoost Regression to predict:

Expected Loss

Predicted Loss

🚨 Fraud Spike Detection

Monthly fraud count

Rolling mean & std

Z-score anomaly tagging

📊 Excel Dashboard

Generated automatically, includes:

Merchant Risk Table

Top 20 High-Risk Merchants

Cluster Summary

Expected vs Predicted Loss Chart

Top Risk Merchant Monthly Anomalies

File generated:
📄 merchant_risk_dashboard.xlsx

🧾 3. Dataset Description

Uses the Sparkov Credit Card Transaction dataset:

File	Description
fraudtrain.csv	Training data (transaction-level)
fraudtest.csv	Test data (transaction-level)

Key columns used:

merchant

amt

category

trans_date_trans_time

is_fraud

🛠 4. Tech Stack
Python Libraries

pandas

numpy

matplotlib

xgboost

scikit-learn

xlsxwriter

Machine Learning

XGBoost Regressor

KMeans Clustering

Analytics

Aggregations

Rolling stats

Z-score anomaly detection

🚀 5. How to Run
Step 1 — Place the datasets

Download & place:

fraudtrain.csv
fraudtest.csv


in the same directory as the script.

Step 2 — Install requirements
pip install pandas numpy scikit-learn xgboost xlsxwriter

Step 3 — Run the script
python export_excel_dashboard.py

Output Generated:

📄 merchant_risk_dashboard.xlsx
Contains 5 sheets with insights & charts.

📈 6. Dashboard Preview
✔ Merchant Scores
✔ Top 20 Risky Merchants
✔ Cluster Summary
✔ Monthly Spike Detection
✔ Loss Prediction Chart

The dashboard is ready for stakeholders, interviews & presentations.

🧮 7. MRI 2.0 Scoring Logic
MRI = 0.45 * fraud_rate
    + 0.15 * dispute_rate
    + 0.10 * scaled_avg_amt
    + 0.10 * scaled_std_amt
    + 0.10 * high_value_rate
    + 0.10 * category_risk


Final score normalized to 0–1 and bucketed into:

Low

Medium

High

📂 8. Folder Structure
Merchant-Risk-Engine/
│
├── fraudtrain.csv
├── fraudtest.csv
├── export_excel_dashboard.py
├── merchant_risk_dashboard.xlsx   (Generated)
└── README.md

⭐ 9. Key Results
Identifies:

Fraud-heavy merchants

High-risk categories

High-value transaction abuse

Abnormal transaction spikes

Merchants contributing highest predicted financial loss

Useful for:

American Express (Credit & Fraud Analytics)

Visa / Mastercard Merchant Risk

Bank Fraud Detection Teams

FinTech Risk & Strategy
