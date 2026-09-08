# ============================================================
# TELECOM CUSTOMER CHURN PREDICTION
# ============================================================
# Case Study:
# Telecom - Classification
#
# Business Problem:
# Acquiring a new customer costs far more than retaining one.
#
# Objective:
# Predict which customers are likely to cancel their subscription.
#
# Analysis Included:
# 1. Data Loading
# 2. Data Cleaning
# 3. Descriptive Analysis
# 4. Diagnostic Analysis
# 5. Predictive Analysis
# 6. Classification Model
# 7. Model Evaluation
# 8. Visualization
# 9. Business Insights
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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. SET FILE PATH
# ============================================================

file_path = r"C:\Users\pravi\PycharmProjects\JupyterProject\Telco_Cust_Churn"

# Check whether file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        "CSV file not found. Please check the file path."
    )

print("File found successfully!")
print()


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(file_path)

print("=" * 70)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 70)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 4. BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])


# ============================================================
# 6. CLEAN TOTALCHARGES
# ============================================================
# TotalCharges is stored as object/string in this dataset.
# Some customers have blank values.
# Convert it to numeric.
# Invalid values become NaN.

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nTotalCharges converted to numeric.")


# ============================================================
# 7. HANDLE MISSING TOTALCHARGES
# ============================================================

print("\nMissing TotalCharges after conversion:")
print(df["TotalCharges"].isnull().sum())

# Fill missing TotalCharges with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)


# ============================================================
# 8. CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE RECORDS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")


# ============================================================
# 9. DESCRIPTIVE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# 9.1 Numerical Summary
# ------------------------------------------------------------

print("\nNumerical Summary:")
print(df.describe())


# ------------------------------------------------------------
# 9.2 Churn Distribution
# ------------------------------------------------------------

print("\nCustomer Churn Distribution:")
print(df["Churn"].value_counts())

print("\nCustomer Churn Percentage:")
churn_percentage = (
    df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(churn_percentage)


# ------------------------------------------------------------
# 9.3 Customer Demographics
# ------------------------------------------------------------

print("\nGender Distribution:")
print(df["gender"].value_counts())

print("\nSenior Citizen Distribution:")
print(df["SeniorCitizen"].value_counts())

print("\nPartner Distribution:")
print(df["Partner"].value_counts())

print("\nDependents Distribution:")
print(df["Dependents"].value_counts())


# ------------------------------------------------------------
# 9.4 Service Information
# ------------------------------------------------------------

print("\nInternet Service Distribution:")
print(df["InternetService"].value_counts())

print("\nContract Distribution:")
print(df["Contract"].value_counts())

print("\nPayment Method Distribution:")
print(df["PaymentMethod"].value_counts())


# ------------------------------------------------------------
# 9.5 Numerical Business Metrics
# ------------------------------------------------------------

print("\nAverage Monthly Charges:")
print(round(df["MonthlyCharges"].mean(), 2))

print("\nAverage Total Charges:")
print(round(df["TotalCharges"].mean(), 2))

print("\nAverage Tenure:")
print(round(df["tenure"].mean(), 2))


# ============================================================
# 10. DESCRIPTIVE VISUALIZATIONS
# ============================================================

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------
# 10.1 Churn Count
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10.2 Contract Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract"
)

plt.title("Customers by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10.3 Internet Service Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="InternetService"
)

plt.title("Customers by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10.4 Monthly Charges Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="MonthlyCharges",
    bins=30,
    kde=True
)

plt.title("Distribution of Monthly Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10.5 Tenure Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="tenure",
    bins=30,
    kde=True
)

plt.title("Distribution of Customer Tenure")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# 11. DIAGNOSTIC ANALYSIS
# ============================================================
# Question:
# WHY are customers churning?
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# 11.1 Churn by Contract
# ------------------------------------------------------------

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Contract:")
print(contract_churn.round(2))


# ------------------------------------------------------------
# 11.2 Churn by Internet Service
# ------------------------------------------------------------

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Internet Service:")
print(internet_churn.round(2))


# ------------------------------------------------------------
# 11.3 Churn by Payment Method
# ------------------------------------------------------------

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Payment Method:")
print(payment_churn.round(2))


# ------------------------------------------------------------
# 11.4 Churn by Senior Citizen
# ------------------------------------------------------------

senior_churn = pd.crosstab(
    df["SeniorCitizen"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Senior Citizen:")
print(senior_churn.round(2))


# ------------------------------------------------------------
# 11.5 Churn by Partner
# ------------------------------------------------------------

partner_churn = pd.crosstab(
    df["Partner"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Partner:")
print(partner_churn.round(2))


# ------------------------------------------------------------
# 11.6 Churn by Dependents
# ------------------------------------------------------------

dependent_churn = pd.crosstab(
    df["Dependents"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Dependents:")
print(dependent_churn.round(2))


# ============================================================
# 12. DIAGNOSTIC VISUALIZATIONS
# ============================================================


# ------------------------------------------------------------
# 12.1 Contract vs Churn
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 12.2 Internet Service vs Churn
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 12.3 Payment Method vs Churn
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn"
)

plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")

plt.xticks(rotation=20)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 12.4 Tenure vs Churn
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure Distribution by Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 12.5 Monthly Charges vs Churn
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 12.6 Total Charges vs Churn
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="TotalCharges"
)

plt.title("Total Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Total Charges")

plt.tight_layout()
plt.show()


# ============================================================
# 13. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

# Convert Churn to numeric temporarily
df["Churn_numeric"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

numeric_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn_numeric"
]

correlation_matrix = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))


# ------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# Remove temporary column
df.drop("Churn_numeric", axis=1, inplace=True)


# ============================================================
# 14. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

print("\n" + "=" * 70)
print("PREDICTIVE ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# Target Variable
# ------------------------------------------------------------

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ------------------------------------------------------------
# Remove Customer ID
# ------------------------------------------------------------

# customerID is only an identifier.
# It should not be used for prediction.

X = df.drop(
    columns=["customerID", "Churn"]
)

y = df["Churn"]


# ============================================================
# 15. IDENTIFY NUMERICAL & CATEGORICAL FEATURES
# ============================================================

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# ============================================================
# 16. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])


# ============================================================
# 17. PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 18. LOGISTIC REGRESSION MODEL
# ============================================================
# Logistic Regression is appropriate because:
# - This is a binary classification problem.
# - Target = Churn / No Churn.
# - It provides probabilities.
# - It is relatively easy to interpret.

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ============================================================
# 19. CREATE MACHINE LEARNING PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


# ============================================================
# 20. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# 21. MAKE PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 22. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)
print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# ============================================================
# 23. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Churn",
            "Churn"
        ]
    )
)


# ============================================================
# 24. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# Confusion Matrix Visualization
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "No Churn",
        "Churn"
    ],
    yticklabels=[
        "No Churn",
        "Churn"
    ]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# 25. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 26. PREDICT CHURN PROBABILITY FOR ALL CUSTOMERS
# ============================================================

df_prediction = df.copy()

# Convert target back to Yes/No for reporting
df_prediction["Actual_Churn"] = df_prediction["Churn"].map({
    0: "No",
    1: "Yes"
})


# Predict probability for all customers
all_features = df.drop(
    columns=["customerID", "Churn"]
)

df_prediction["Churn_Probability"] = (
    pipeline.predict_proba(all_features)[:, 1]
)


# Convert probability into percentage
df_prediction["Churn_Probability_%"] = (
    df_prediction["Churn_Probability"] * 100
).round(2)


# ============================================================
# 27. CREATE CHURN RISK CATEGORY
# ============================================================

def classify_risk(probability):

    if probability >= 0.70:
        return "High Risk"

    elif probability >= 0.40:
        return "Medium Risk"

    else:
        return "Low Risk"


df_prediction["Risk_Category"] = (
    df_prediction["Churn_Probability"]
    .apply(classify_risk)
)


# ============================================================
# 28. CUSTOMER CHURN RISK SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER CHURN RISK SUMMARY")
print("=" * 70)

risk_summary = (
    df_prediction["Risk_Category"]
    .value_counts()
)

print(risk_summary)


# ============================================================
# 29. RISK CATEGORY VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df_prediction,
    x="Risk_Category",
    order=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)

plt.title("Customer Churn Risk Categories")
plt.xlabel("Risk Category")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# 30. TOP HIGH-RISK CUSTOMERS
# ============================================================

high_risk_customers = (
    df_prediction[
        df_prediction["Risk_Category"] == "High Risk"
    ]
    .sort_values(
        by="Churn_Probability",
        ascending=False
    )
)


print("\n" + "=" * 70)
print("TOP HIGH-RISK CUSTOMERS")
print("=" * 70)

columns_to_display = [
    "customerID",
    "tenure",
    "Contract",
    "InternetService",
    "MonthlyCharges",
    "TotalCharges",
    "PaymentMethod",
    "Actual_Churn",
    "Churn_Probability_%",
    "Risk_Category"
]

print(
    high_risk_customers[
        columns_to_display
    ].head(20)
)


# ============================================================
# 31. EXPORT PREDICTIONS
# ============================================================

output_file = "telecom_customer_churn_predictions.csv"

df_prediction[
    columns_to_display
].to_csv(
    output_file,
    index=False
)

print("\nPrediction file created:")
print(output_file)


# ============================================================
# 32. BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS INSIGHTS")
print("=" * 70)

print("""
1. Customer churn is a major business risk because losing an
   existing customer creates additional acquisition costs.

2. Contract type is an important churn indicator. Customers
   on shorter contracts generally require greater retention
   attention.

3. Customer tenure is useful for identifying customers who
   may be at higher risk, especially newer customers.

4. Monthly charges can be associated with churn behavior.
   Customers with higher monthly charges may require targeted
   retention offers.

5. Payment method can also reveal differences in churn behavior.

6. Internet service type can help identify customer segments
   that require additional retention strategies.

7. The predictive model assigns each customer a probability
   of churn.

8. High-risk customers can be prioritized for retention campaigns.

9. Instead of giving the same offer to every customer, the
   telecom company can target customers based on their
   predicted churn probability.

10. Retaining high-risk existing customers can potentially
    reduce customer acquisition costs.
""")


# ============================================================
# 33. FINAL MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
Dataset Size:
    {df.shape[0]} customers
    {df.shape[1]} variables

Model:
    Logistic Regression

Accuracy:
    {accuracy:.2%}

Precision:
    {precision:.2%}

Recall:
    {recall:.2%}

F1 Score:
    {f1:.2%}

ROC-AUC:
    {roc_auc:.2%}

High-Risk Customers:
    {risk_summary.get("High Risk", 0)}

Medium-Risk Customers:
    {risk_summary.get("Medium Risk", 0)}

Low-Risk Customers:
    {risk_summary.get("Low Risk", 0)}

Business Objective:
    Identify customers with a high probability of churn
    so that the company can take preventive retention action.
""")

print("=" * 70)
print("TELECOM CUSTOMER CHURN ANALYSIS COMPLETED")
print("=" * 70)