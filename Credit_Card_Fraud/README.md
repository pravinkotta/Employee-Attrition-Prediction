# Credit Card Fraud Detection

## 📌 Project Overview

Credit card fraud detection is a critical problem in the banking and financial services industry.

Every missed fraudulent transaction can result in a direct financial loss, while every false fraud alert can inconvenience a genuine customer.

This project uses **Machine Learning classification** to identify potentially fraudulent credit card transactions.

The project performs:

* Descriptive Analytics
* Diagnostic Analytics
* Predictive Analytics
* Exploratory Data Analysis
* Data preprocessing
* Data visualization
* Classification modeling
* Fraud probability prediction
* Risk categorization
* Model evaluation
* Business insights

---

## 🎯 Business Problem

> **"Every missed fraud is a direct loss; every false alarm annoys a genuine customer."**

The objective is to develop a machine learning model that can identify fraudulent transactions while considering the highly imbalanced nature of credit card transaction data.

The model should help answer:

* Which transactions are potentially fraudulent?
* How frequently does fraud occur?
* What transaction characteristics are associated with fraud?
* How accurately can fraudulent transactions be detected?
* How many genuine transactions are incorrectly flagged?
* How many fraudulent transactions are missed?
* Which transactions should be considered high risk?

---

## 🎯 Project Objective

The main objective of this project is to build a **Credit Card Fraud Detection Classification Model**.

The workflow includes:

1. Load the credit card transaction dataset.
2. Perform data quality checks.
3. Analyze the distribution of fraudulent and legitimate transactions.
4. Perform descriptive analytics.
5. Perform diagnostic analytics.
6. Visualize important patterns.
7. Prepare the data for machine learning.
8. Handle the highly imbalanced target variable.
9. Train a Logistic Regression classification model.
10. Generate fraud probabilities.
11. Categorize transactions into risk levels.
12. Evaluate model performance using appropriate classification metrics.
13. Analyze false positives and false negatives.
14. Generate business insights.
15. Export transaction-level predictions.

---

# 📊 Dataset

## Dataset Source

The dataset is obtained from Kaggle:

**Credit Card Fraud Detection — Machine Learning Group (ULB)**

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The original Kaggle dataset contains anonymized credit card transactions made by European cardholders during September 2013. It contains **284,807 transactions**, including **492 fraudulent transactions**. Fraud represents approximately **0.172%** of all transactions.

### Dataset Characteristics

| Property                |            Value |
| ----------------------- | ---------------: |
| Total Transactions      |          284,807 |
| Fraudulent Transactions |              492 |
| Legitimate Transactions |          284,315 |
| Fraud Percentage        |          ~0.172% |
| Number of Columns       |               31 |
| Target Variable         |          `Class` |
| Dataset File            | `creditcard.csv` |
| Dataset Size            |          ~150 MB |

The dataset contains 28 anonymized PCA-derived variables (`V1`–`V28`). The `Time` and `Amount` variables were not transformed using PCA, while `Class` is the target variable.

---

# 📁 Dataset Columns

The dataset contains:

```text
Time
V1
V2
V3
...
V28
Amount
Class
```

### Important Variables

### `Time`

Number of seconds elapsed between the transaction and the first transaction in the dataset.

### `V1` – `V28`

Anonymized features obtained through PCA transformation.

The original transaction features are not publicly available because of confidentiality considerations.

### `Amount`

Transaction amount.

### `Class`

Target variable:

```text
0 = Legitimate Transaction
1 = Fraudulent Transaction
```

---

# ⚠️ Class Imbalance

One of the most important characteristics of this dataset is its extreme class imbalance.

There are:

```text
284,315 legitimate transactions
492 fraudulent transactions
```

Therefore, fraudulent transactions represent only approximately **0.172%** of the dataset.

This means that **accuracy alone is not a reliable metric**.

For example, a model that predicts almost every transaction as legitimate could achieve very high accuracy while failing to detect most fraud cases.

Therefore, this project focuses on:

* Precision
* Recall
* F1 Score
* ROC-AUC
* Precision-Recall AUC / PR-AUC
* Confusion Matrix

Kaggle specifically recommends using the **Area Under the Precision-Recall Curve (AUPRC)** for this dataset because of the severe class imbalance.

---

# 🔍 Analytics Performed

## 1. Descriptive Analytics

Descriptive analysis is used to understand the existing transaction data.

The project examines:

* Number of transactions
* Number of features
* Data types
* Missing values
* Duplicate records
* Statistical summary
* Transaction amounts
* Fraud distribution
* Average transaction amount
* Minimum transaction amount
* Maximum transaction amount

Example analysis:

```python
df.describe()
```

---

# 🔎 2. Diagnostic Analytics

Diagnostic analytics investigates patterns that may help explain fraudulent transactions.

The project analyzes:

* Fraud vs legitimate transactions
* Transaction amount patterns
* Transaction time patterns
* Fraud rate by amount group
* Fraud rate by time group
* Correlation between features
* Relationship between features and fraud

Visualizations are used to identify potentially important patterns.

---

# 🤖 3. Predictive Analytics

A Machine Learning classification model is developed to predict whether a transaction is:

```text
Legitimate
```

or

```text
Fraudulent
```

The project uses:

### Logistic Regression

Logistic Regression is used as the baseline classification model.

The model produces a probability indicating how likely a transaction is to be fraudulent.

Example:

```text
Fraud Probability = 0.82
```

This transaction can be considered high risk.

---

# ⚙️ Machine Learning Workflow

The project follows the following workflow:

```text
Raw Dataset
     ↓
Data Loading
     ↓
Data Quality Checks
     ↓
Duplicate Handling
     ↓
Exploratory Data Analysis
     ↓
Descriptive Analytics
     ↓
Diagnostic Analytics
     ↓
Feature / Target Separation
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Logistic Regression
     ↓
Fraud Probability
     ↓
Classification
     ↓
Model Evaluation
     ↓
Risk Categorization
     ↓
Business Insights
```

---

# 🧹 Data Preprocessing

The following preprocessing steps are performed:

### 1. Missing Value Check

```python
df.isnull().sum()
```

### 2. Duplicate Check

Duplicate transactions are identified and removed where appropriate.

### 3. Feature and Target Separation

```text
X = Features
y = Class
```

### 4. Stratified Train-Test Split

A stratified split is used so that both training and testing datasets preserve the fraud/legitimate class distribution.

### 5. Feature Scaling

`Time` and `Amount` are standardized using `StandardScaler`.

---

# ⚖️ Handling Class Imbalance

The Logistic Regression model uses:

```python
class_weight="balanced"
```

This gives greater importance to the minority fraud class during model training.

This is particularly important because the dataset contains only 492 fraudulent transactions compared with 284,315 legitimate transactions.

---

# 📈 Model Evaluation

The following metrics are calculated:

## Accuracy

Percentage of correctly classified transactions.

However, accuracy is not considered the primary metric because of the extreme class imbalance.

---

## Precision

Of all transactions predicted as fraud, precision measures how many were actually fraudulent.

```text
Precision =
True Positives / (True Positives + False Positives)
```

High precision means fewer legitimate customers are incorrectly flagged.

---

## Recall

Recall measures how many actual fraudulent transactions were successfully detected.

```text
Recall =
True Positives / (True Positives + False Negatives)
```

For fraud detection, recall is especially important because missing a fraudulent transaction can result in financial loss.

---

## F1 Score

F1 Score provides a balance between precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## ROC-AUC

ROC-AUC measures the model's ability to distinguish between legitimate and fraudulent transactions across classification thresholds.

---

## PR-AUC

Precision-Recall AUC is especially important for this project because the fraud class is extremely rare.

Kaggle recommends AUPRC as a meaningful evaluation measure for this dataset.

---

# 📊 Visualizations

The project generates several visualizations, including:

### Class Distribution

Shows the difference between:

* Legitimate transactions
* Fraudulent transactions

### Transaction Amount Distribution

Shows how transaction amounts are distributed.

### Fraud vs Legitimate Amount Distribution

Compares transaction amounts between the two classes.

### Transaction Time Distribution

Examines transaction activity over time.

### Fraud Rate by Amount Group

Shows how fraud rates vary across transaction amount ranges.

### Fraud Rate by Time Group

Examines fraud rates across transaction time periods.

### Correlation Heatmap

Shows relationships between numerical features and the target variable.

### Confusion Matrix

Shows:

```text
True Positive
True Negative
False Positive
False Negative
```

### ROC Curve

Visualizes the model's classification performance.

### Precision-Recall Curve

Visualizes the trade-off between precision and recall.

### Logistic Regression Coefficients

Shows the features contributing most strongly to the model's predictions.

---

# 🚨 Fraud Risk Categorization

The model generates a fraud probability for every transaction.

Transactions are then categorized into risk levels.

| Fraud Probability | Risk Level  |
| ----------------: | ----------- |
|          `< 0.40` | Low Risk    |
|   `0.40 – < 0.70` | Medium Risk |
|         `>= 0.70` | High Risk   |

Example:

```text
Fraud Probability = 0.15
Risk = Low

Fraud Probability = 0.55
Risk = Medium

Fraud Probability = 0.91
Risk = High
```

These thresholds are used for analytical demonstration and can be adjusted depending on business requirements.

---

# ❌ False Positive and False Negative Analysis

Fraud detection requires balancing two types of errors.

## False Positive

A legitimate transaction is incorrectly classified as fraud.

```text
Actual = Legitimate
Predicted = Fraud
```

Business impact:

* Customer inconvenience
* Transaction rejection
* Additional verification
* Customer dissatisfaction

---

## False Negative

A fraudulent transaction is incorrectly classified as legitimate.

```text
Actual = Fraud
Predicted = Legitimate
```

Business impact:

* Financial loss
* Customer loss
* Increased fraud exposure
* Potential regulatory and operational costs

Therefore, fraud detection models must carefully balance precision and recall.

---

# 💼 Business Insights

The analysis can help financial institutions:

### 1. Identify High-Risk Transactions

Transactions with high predicted fraud probabilities can be prioritized for investigation.

### 2. Reduce Financial Loss

Early identification of suspicious transactions can help reduce successful fraudulent transactions.

### 3. Reduce Customer Friction

Improving precision can reduce the number of legitimate transactions incorrectly blocked.

### 4. Prioritize Manual Investigation

Risk scores can help fraud teams focus their attention on the highest-risk transactions.

### 5. Support Real-Time Fraud Detection

The modeling approach demonstrates how transaction-level predictions could form part of a larger real-time fraud detection system.

---

# 📤 Output File

The project generates:

```text
credit_card_fraud_predictions.csv
```

The output contains transaction-level predictions and risk information.

Typical prediction columns include:

```text
Actual_Class
Predicted_Class
Fraud_Probability
Risk_Category
```

The output file is saved in the same folder as the Jupyter Notebook.

---

# 📂 Project Structure

```text
Credit_Card_Fraud/
│
├── Credit_Card_Fraud_Detection.ipynb
│
├── creditcard.csv
│
├── credit_card_fraud_predictions.csv
│
└── README.md
```

> **Note:** The original `creditcard.csv` file is approximately 150 MB. It is recommended not to upload the raw dataset to GitHub. Instead, provide the Kaggle dataset source link in the README.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

### Development Environment

* Jupyter Notebook
* PyCharm

### Dataset Platform

* Kaggle

---

# 📦 Installation

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# ▶️ How to Run the Project

### Step 1 — Download the Dataset

Download `creditcard.csv` from the Kaggle dataset:

**Credit Card Fraud Detection**

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Step 2 — Place the Dataset

Place:

```text
creditcard.csv
```

in the same folder as the Jupyter Notebook.

### Step 3 — Open the Notebook

Open:

```text
Credit_Card_Fraud_Detection.ipynb
```

using Jupyter Notebook or PyCharm.

### Step 4 — Run the Notebook

Run the cells from top to bottom.

The notebook will:

```text
Load Dataset
      ↓
Perform Data Analysis
      ↓
Generate Visualizations
      ↓
Train Machine Learning Model
      ↓
Evaluate Model
      ↓
Generate Fraud Probabilities
      ↓
Create Risk Categories
      ↓
Export Predictions
```

---

# 📌 Dataset Citation

The dataset used in this project was created through a research collaboration involving **Worldline** and the **Machine Learning Group of ULB (Université Libre de Bruxelles)** and is distributed through Kaggle.

### Kaggle Dataset

**Machine Learning Group - ULB**

Credit Card Fraud Detection

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The Kaggle dataset page provides the dataset description, provenance, licensing information, and recommended evaluation considerations.

---

# 📚 References

The Kaggle dataset cites research including:

* Andrea Dal Pozzolo et al., *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE Symposium on Computational Intelligence and Data Mining, 2015.
* Andrea Dal Pozzolo et al., *Learned lessons in credit card fraud detection from a practitioner perspective*, Expert Systems with Applications, 2014.
* Andrea Dal Pozzolo et al., *Credit card fraud detection: a realistic modeling and a novel learning strategy*, IEEE Transactions on Neural Networks and Learning Systems, 2018.

The complete citation information is available on the Kaggle dataset page.

---

# 👨‍💻 Project Type

**AI Engineering / Machine Learning Portfolio Project**

### Domain

Banking & Financial Services

### Problem Type

Classification

### Analytics

* Descriptive Analytics
* Diagnostic Analytics
* Predictive Analytics

### Machine Learning

Logistic Regression

### Key Challenge

Extreme class imbalance in fraudulent transactions.

---

# ⭐ Key Learning Outcomes

Through this project, the following concepts are demonstrated:

* Loading large CSV datasets
* Data quality analysis
* Exploratory Data Analysis
* Descriptive statistics
* Data visualization
* Correlation analysis
* Feature engineering/preprocessing
* Feature scaling
* Stratified train-test splitting
* Imbalanced classification
* Logistic Regression
* Probability prediction
* Confusion matrix
* Precision
* Recall
* F1 Score
* ROC-AUC
* Precision-Recall AUC
* Risk categorization
* False positive analysis
* False negative analysis
* Business interpretation of ML results
* Exporting model predictions

---

# ⚠️ Disclaimer

This project is intended for **educational and portfolio purposes**.

The model should not be considered a production-ready banking fraud detection system. Real-world fraud detection systems require additional considerations such as real-time transaction streams, model monitoring, threshold optimization, concept drift, cost-sensitive learning, security, regulatory requirements, and continuous model retraining.

---

## 📌 Author

**Pravin**

AI Engineering Program
Machine Learning | Data Analytics | Python
