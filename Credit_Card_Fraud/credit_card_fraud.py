# ============================================================
# CREDIT CARD FRAUD DETECTION
# Case Study 2 - Banking | Classification
#
# Descriptive + Diagnostic + Predictive Analysis
# With Visualizations
#
# Dataset:
# Kaggle - Credit Card Fraud Detection
#
# File:
# creditcard.csv
#
# Output:
# credit_card_fraud_predictions.csv
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

warnings.filterwarnings("ignore")

print("Libraries imported successfully.")


# ============================================================
# 2. SET PROJECT FOLDER
# ============================================================

# Get the current Jupyter Notebook working directory
project_folder = os.getcwd()

print("\nCurrent project folder:")
print(project_folder)

# Dataset file is in the same folder as the notebook
file_path = os.path.join(
    project_folder,
    "creditcard.csv"
)

print("\nDataset path:")
print(file_path)


# ============================================================
# 3. LOAD DATASET
# ============================================================

if not os.path.exists(file_path):

    raise FileNotFoundError(
        "\ncreditcard.csv was not found.\n\n"
        "Expected location:\n"
        + file_path +
        "\n\n"
        "Please place creditcard.csv in the same folder "
        "as your Jupyter Notebook."
    )

df = pd.read_csv(file_path)

print("\nDataset loaded successfully.")

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 4. BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("BASIC DATA INFORMATION")
print("=" * 60)

print("\nFirst 5 rows:")
display(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
display(df.dtypes)

print("\nDataset information:")
df.info()


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()

print("\nMissing values by column:")

display(
    missing_values[missing_values > 0]
)

if missing_values.sum() == 0:

    print("\nNo missing values found.")

else:

    print(
        f"\nTotal missing values: "
        f"{missing_values.sum():,}"
    )


# ============================================================
# 6. CHECK DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE RECORD ANALYSIS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print(
    f"\nNumber of duplicate rows: "
    f"{duplicate_count:,}"
)


# ============================================================
# 7. REMOVE DUPLICATE RECORDS
# ============================================================

if duplicate_count > 0:

    df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    print("\nDuplicate records removed.")

else:

    print("\nNo duplicate records found.")

print("\nCurrent dataset shape:")
print(df.shape)


# ============================================================
# 8. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

display(
    df.describe().T
)


# ============================================================
# 9. FRAUD DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("FRAUD DISTRIBUTION")
print("=" * 60)

class_counts = df["Class"].value_counts()

print("\nTransaction counts:")
print(class_counts)

class_percentages = (
    df["Class"]
    .value_counts(normalize=True)
    * 100
)

print("\nTransaction percentages:")
print(class_percentages)

legitimate_count = class_counts.get(0, 0)

fraud_count = class_counts.get(1, 0)

fraud_percentage = (
    fraud_count / len(df)
) * 100

print("\nLegitimate transactions:")
print(f"{legitimate_count:,}")

print("\nFraudulent transactions:")
print(f"{fraud_count:,}")

print("\nFraud percentage:")
print(f"{fraud_percentage:.4f}%")


# ============================================================
# 10. FRAUD AMOUNT ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("TRANSACTION AMOUNT ANALYSIS")
print("=" * 60)

legitimate_amount = (
    df[df["Class"] == 0]["Amount"]
)

fraud_amount = (
    df[df["Class"] == 1]["Amount"]
)

print("\nLegitimate transaction amount:")
display(
    legitimate_amount.describe()
)

print("\nFraudulent transaction amount:")
display(
    fraud_amount.describe()
)

print("\nAverage legitimate transaction amount:")
print(
    f"{legitimate_amount.mean():.2f}"
)

print("\nAverage fraudulent transaction amount:")
print(
    f"{fraud_amount.mean():.2f}"
)

print("\nMedian legitimate transaction amount:")
print(
    f"{legitimate_amount.median():.2f}"
)

print("\nMedian fraudulent transaction amount:")
print(
    f"{fraud_amount.median():.2f}"
)


# ============================================================
# 11. TRANSACTION TIME ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("TRANSACTION TIME ANALYSIS")
print("=" * 60)

legitimate_time = (
    df[df["Class"] == 0]["Time"]
)

fraud_time = (
    df[df["Class"] == 1]["Time"]
)

print("\nAverage transaction time:")

print(
    f"Legitimate: "
    f"{legitimate_time.mean():.2f} seconds"
)

print(
    f"Fraudulent: "
    f"{fraud_time.mean():.2f} seconds"
)

print("\nMedian transaction time:")

print(
    f"Legitimate: "
    f"{legitimate_time.median():.2f} seconds"
)

print(
    f"Fraudulent: "
    f"{fraud_time.median():.2f} seconds"
)


# ============================================================
# 12. CLASS DISTRIBUTION VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Class"
)

plt.title(
    "Credit Card Transaction Class Distribution"
)

plt.xlabel(
    "Class (0 = Legitimate, 1 = Fraud)"
)

plt.ylabel(
    "Number of Transactions"
)

plt.tight_layout()

plt.show()


# ============================================================
# 13. TRANSACTION AMOUNT DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Amount",
    bins=100
)

plt.xscale("log")

plt.title(
    "Transaction Amount Distribution"
)

plt.xlabel(
    "Transaction Amount - Log Scale"
)

plt.ylabel(
    "Number of Transactions"
)

plt.tight_layout()

plt.show()


# ============================================================
# 14. FRAUD VS LEGITIMATE TRANSACTION AMOUNT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.yscale("log")

plt.title(
    "Transaction Amount by Class"
)

plt.xlabel(
    "Class (0 = Legitimate, 1 = Fraud)"
)

plt.ylabel(
    "Transaction Amount - Log Scale"
)

plt.tight_layout()

plt.show()


# ============================================================
# 15. TRANSACTION TIME DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Time",
    bins=100
)

plt.title(
    "Transaction Time Distribution"
)

plt.xlabel(
    "Time (Seconds)"
)

plt.ylabel(
    "Number of Transactions"
)

plt.tight_layout()

plt.show()


# ============================================================
# 16. FRAUD RATE BY TRANSACTION AMOUNT
# ============================================================

df["Amount_Bin"] = pd.qcut(
    df["Amount"],
    q=10,
    duplicates="drop"
)

amount_fraud_rate = (
    df.groupby(
        "Amount_Bin",
        observed=True
    )["Class"]
    .mean()
    * 100
)

print("\n" + "=" * 60)
print("FRAUD RATE BY TRANSACTION AMOUNT")
print("=" * 60)

display(
    amount_fraud_rate
)


# ============================================================
# 17. FRAUD RATE BY TRANSACTION TIME
# ============================================================

df["Time_Bin"] = pd.qcut(
    df["Time"],
    q=10,
    duplicates="drop"
)

time_fraud_rate = (
    df.groupby(
        "Time_Bin",
        observed=True
    )["Class"]
    .mean()
    * 100
)

print("\n" + "=" * 60)
print("FRAUD RATE BY TRANSACTION TIME")
print("=" * 60)

display(
    time_fraud_rate
)


# ============================================================
# 18. FRAUD RATE BY AMOUNT GROUP VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 5))

amount_fraud_rate.plot(
    kind="bar"
)

plt.title(
    "Fraud Rate by Transaction Amount Group"
)

plt.xlabel(
    "Transaction Amount Group"
)

plt.ylabel(
    "Fraud Rate (%)"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.show()


# ============================================================
# 19. FRAUD RATE BY TIME GROUP VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 5))

time_fraud_rate.plot(
    kind="bar"
)

plt.title(
    "Fraud Rate by Transaction Time Group"
)

plt.xlabel(
    "Transaction Time Group"
)

plt.ylabel(
    "Fraud Rate (%)"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.show()


# ============================================================
# 20. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

correlation_df = df.drop(
    columns=[
        "Amount_Bin",
        "Time_Bin"
    ],
    errors="ignore"
)

correlation = (
    correlation_df.corr()
)

class_correlation = (
    correlation["Class"]
    .drop("Class")
    .sort_values(
        ascending=False
    )
)

print(
    "\nFeatures positively correlated "
    "with fraud:"
)

display(
    class_correlation.head(10)
)

print(
    "\nFeatures negatively correlated "
    "with fraud:"
)

display(
    class_correlation.tail(10)
)


# ============================================================
# 21. CORRELATION HEATMAP
# ============================================================

plt.figure(
    figsize=(16, 12)
)

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    linewidths=0.1
)

plt.title(
    "Credit Card Fraud Detection - Correlation Heatmap"
)

plt.tight_layout()

plt.show()


# ============================================================
# 22. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

df_model = df.drop(
    columns=[
        "Amount_Bin",
        "Time_Bin"
    ],
    errors="ignore"
).copy()

print("\n" + "=" * 60)
print("MODEL DATASET")
print("=" * 60)

print("\nModel dataset shape:")
print(df_model.shape)


# ============================================================
# 23. SEPARATE FEATURES AND TARGET
# ============================================================

X = df_model.drop(
    "Class",
    axis=1
)

y = df_model["Class"]

print("\n" + "=" * 60)
print("FEATURES AND TARGET")
print("=" * 60)

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 24. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nFraud cases in training data:")
print(y_train.sum())

print("\nFraud cases in testing data:")
print(y_test.sum())


# ============================================================
# 25. FEATURE SCALING
# ============================================================

X_train_scaled = X_train.copy()

X_test_scaled = X_test.copy()

scaler = StandardScaler()

X_train_scaled[
    ["Time", "Amount"]
] = scaler.fit_transform(
    X_train[
        ["Time", "Amount"]
    ]
)

X_test_scaled[
    ["Time", "Amount"]
] = scaler.transform(
    X_test[
        ["Time", "Amount"]
    ]
)

print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)

print(
    "\nTime and Amount features scaled successfully."
)


# ============================================================
# 26. CREATE LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION MODEL")
print("=" * 60)

print(
    "\nLogistic Regression model created."
)

print(
    "\nClass weight: balanced"
)


# ============================================================
# 27. TRAIN THE MACHINE LEARNING MODEL
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

print(
    "\nTraining model..."
)

model.fit(
    X_train_scaled,
    y_train
)

print(
    "Model training completed successfully."
)


# ============================================================
# 28. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test_scaled
)

y_pred_probability = (
    model.predict_proba(
        X_test_scaled
    )[:, 1]
)

print("\n" + "=" * 60)
print("PREDICTIONS")
print("=" * 60)

print(
    "\nPredictions generated successfully."
)


# ============================================================
# 29. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_pred_probability
)

pr_auc = average_precision_score(
    y_test,
    y_pred_probability
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)


# ============================================================
# 30. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# 31. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Legitimate",
        "Fraud"
    ],
    yticklabels=[
        "Legitimate",
        "Fraud"
    ]
)

plt.title(
    "Credit Card Fraud Detection - Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()

plt.show()


# ============================================================
# 32. ROC CURVE
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    y_pred_probability
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title(
    "ROC Curve"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 33. PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_test,
        y_pred_probability
    )
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall_curve,
    precision_curve,
    label=f"PR-AUC = {pr_auc:.4f}"
)

plt.title(
    "Precision-Recall Curve"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 34. CREATE TEST RESULTS DATAFRAME
# ============================================================

results = X_test.copy()

results["Actual_Class"] = (
    y_test.values
)

results["Predicted_Class"] = (
    y_pred
)

results["Fraud_Probability"] = (
    y_pred_probability
)

results = results.sort_values(
    by="Fraud_Probability",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP FRAUD PROBABILITY TRANSACTIONS")
print("=" * 60)

display(
    results[
        [
            "Time",
            "Amount",
            "Actual_Class",
            "Predicted_Class",
            "Fraud_Probability"
        ]
    ].head(20)
)


# ============================================================
# 35. CREATE FRAUD RISK CATEGORIES
# ============================================================

def fraud_risk(probability):

    if probability >= 0.70:

        return "High Risk"

    elif probability >= 0.40:

        return "Medium Risk"

    else:

        return "Low Risk"


results["Fraud_Risk"] = (
    results[
        "Fraud_Probability"
    ].apply(
        fraud_risk
    )
)

print("\n" + "=" * 60)
print("FRAUD RISK CATEGORIES")
print("=" * 60)

display(
    results[
        [
            "Time",
            "Amount",
            "Actual_Class",
            "Predicted_Class",
            "Fraud_Probability",
            "Fraud_Risk"
        ]
    ].head(20)
)


# ============================================================
# 36. FRAUD RISK SUMMARY
# ============================================================

risk_summary = (
    results["Fraud_Risk"]
    .value_counts()
    .reindex(
        [
            "High Risk",
            "Medium Risk",
            "Low Risk"
        ],
        fill_value=0
    )
)

print("\n" + "=" * 60)
print("FRAUD RISK SUMMARY")
print("=" * 60)

display(
    risk_summary
)


# ============================================================
# 37. FRAUD RISK VISUALIZATION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.countplot(
    data=results,
    x="Fraud_Risk",
    order=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)

plt.title(
    "Predicted Fraud Risk Categories"
)

plt.xlabel(
    "Fraud Risk"
)

plt.ylabel(
    "Number of Transactions"
)

plt.tight_layout()

plt.show()


# ============================================================
# 38. IDENTIFY HIGH-RISK TRANSACTIONS
# ============================================================

high_risk_transactions = (
    results[
        results["Fraud_Risk"]
        == "High Risk"
    ]
    .copy()
)

print("\n" + "=" * 60)
print("HIGH-RISK TRANSACTIONS")
print("=" * 60)

print(
    f"\nNumber of high-risk transactions: "
    f"{len(high_risk_transactions):,}"
)

display(
    high_risk_transactions[
        [
            "Time",
            "Amount",
            "Actual_Class",
            "Predicted_Class",
            "Fraud_Probability",
            "Fraud_Risk"
        ]
    ].head(20)
)


# ============================================================
# 39. ACTUAL FRAUD DETECTION PERFORMANCE
# ============================================================

actual_fraud = (
    results[
        results["Actual_Class"] == 1
    ]
)

detected_fraud = (
    actual_fraud[
        actual_fraud["Predicted_Class"] == 1
    ]
)

missed_fraud = (
    actual_fraud[
        actual_fraud["Predicted_Class"] == 0
    ]
)

print("\n" + "=" * 60)
print("ACTUAL FRAUD DETECTION PERFORMANCE")
print("=" * 60)

print(
    f"\nActual fraud cases   : "
    f"{len(actual_fraud):,}"
)

print(
    f"Detected fraud cases : "
    f"{len(detected_fraud):,}"
)

print(
    f"Missed fraud cases   : "
    f"{len(missed_fraud):,}"
)

if len(actual_fraud) > 0:

    detection_rate = (
        len(detected_fraud)
        / len(actual_fraud)
        * 100
    )

else:

    detection_rate = 0


print(
    f"\nFraud detection rate : "
    f"{detection_rate:.2f}%"
)


# ============================================================
# 40. FALSE POSITIVE / FALSE NEGATIVE ANALYSIS
# ============================================================

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 60)
print("CONFUSION MATRIX BUSINESS ANALYSIS")
print("=" * 60)

print(
    f"\nTrue Negatives  : {tn:,}"
)

print(
    f"False Positives : {fp:,}"
)

print(
    f"False Negatives : {fn:,}"
)

print(
    f"True Positives  : {tp:,}"
)

print("\nBusiness meaning:")

print(
    "\nFalse Negatives = Fraudulent transactions "
    "that were missed."
)

print(
    "False Positives = Genuine transactions "
    "incorrectly flagged as fraud."
)

print(
    "True Positives = Fraudulent transactions "
    "correctly detected."
)

print(
    "True Negatives = Genuine transactions "
    "correctly identified."
)


# ============================================================
# 41. FEATURE IMPORTANCE / MODEL COEFFICIENTS
# ============================================================

feature_coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

feature_coefficients[
    "Absolute_Coefficient"
] = feature_coefficients[
    "Coefficient"
].abs()

feature_coefficients = (
    feature_coefficients
    .sort_values(
        by="Absolute_Coefficient",
        ascending=False
    )
)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION FEATURE COEFFICIENTS")
print("=" * 60)

display(
    feature_coefficients.head(15)
)


# ============================================================
# 42. COEFFICIENT VISUALIZATION
# ============================================================

top_coefficients = (
    feature_coefficients
    .head(15)
    .sort_values(
        by="Coefficient"
    )
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_coefficients["Feature"],
    top_coefficients["Coefficient"]
)

plt.title(
    "Top Logistic Regression Feature Coefficients"
)

plt.xlabel(
    "Coefficient"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.show()


# ============================================================
# 43. EXPORT PREDICTIONS
# ============================================================

output_file = os.path.join(
    project_folder,
    "credit_card_fraud_predictions.csv"
)

results.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("OUTPUT FILE")
print("=" * 60)

print(
    "\nPrediction file created successfully!"
)

print(
    "\nSaved at:"
)

print(
    output_file
)


# ============================================================
# 44. BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

print(
    f"\n1. Total transactions analyzed: "
    f"{len(df):,}"
)

print(
    f"\n2. Actual fraudulent transactions: "
    f"{int(df['Class'].sum()):,}"
)

print(
    f"\n3. Fraud rate: "
    f"{df['Class'].mean() * 100:.4f}%"
)

print(
    f"\n4. Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"\n5. Precision: "
    f"{precision:.4f}"
)

print(
    f"\n6. Recall: "
    f"{recall:.4f}"
)

print(
    f"\n7. F1 Score: "
    f"{f1:.4f}"
)

print(
    f"\n8. ROC-AUC: "
    f"{roc_auc:.4f}"
)

print(
    f"\n9. PR-AUC: "
    f"{pr_auc:.4f}"
)

print(
    f"\n10. High-risk transactions identified: "
    f"{len(high_risk_transactions):,}"
)

print(
    "\nBusiness Interpretation:"
)

print(
    "\n- High recall helps reduce missed fraudulent transactions."
)

print(
    "- High precision helps reduce false alarms for genuine customers."
)

print(
    "- PR-AUC is especially important because fraud "
    "represents a very small percentage of all transactions."
)

print(
    "- High-risk transactions can be prioritized for "
    "additional verification or manual review."
)


# ============================================================
# 45. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CREDIT CARD FRAUD DETECTION")
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print("\nDataset:")
print(
    f"Total Transactions : {len(df):,}"
)

print(
    f"Fraud Cases        : {int(df['Class'].sum()):,}"
)

print(
    f"Fraud Rate         : "
    f"{df['Class'].mean() * 100:.4f}%"
)

print("\nModel:")
print(
    "Logistic Regression"
)

print(
    "Class Weight: Balanced"
)

print("\nPerformance:")

print(
    f"Accuracy     : {accuracy:.4f}"
)

print(
    f"Precision    : {precision:.4f}"
)

print(
    f"Recall       : {recall:.4f}"
)

print(
    f"F1 Score     : {f1:.4f}"
)

print(
    f"ROC-AUC      : {roc_auc:.4f}"
)

print(
    f"PR-AUC       : {pr_auc:.4f}"
)

print("\nBusiness Results:")

print(
    f"Detected Fraud Cases : "
    f"{len(detected_fraud):,}"
)

print(
    f"Missed Fraud Cases   : "
    f"{len(missed_fraud):,}"
)

print(
    f"High-Risk Transactions: "
    f"{len(high_risk_transactions):,}"
)

print("\nOutput File:")

print(
    output_file
)

print(
    "\nCredit Card Fraud Detection project "
    "completed successfully."
)