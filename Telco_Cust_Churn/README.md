# 📊 Telecom Customer Churn Prediction

## 📌 Project Overview

Customer churn is a major challenge for telecom companies. Acquiring a new customer can cost significantly more than retaining an existing customer.

This project uses **data analytics and machine learning** to understand customer churn and predict which customers are most likely to cancel their telecom subscription.

The project follows three major stages of analytics:

* **Descriptive Analysis** – What is happening?
* **Diagnostic Analysis** – Why is it happening?
* **Predictive Analysis** – Which customers are likely to churn?

The final machine learning model assigns each customer a **churn probability** and categorizes customers into **Low, Medium, and High Risk** groups.

---

## 🎯 Business Problem

### Problem Statement

Telecom companies lose revenue when customers cancel their subscriptions.

The business needs to identify customers who are likely to churn **before they leave**, so that appropriate retention strategies can be applied.

### Business Objective

> **Predict which customers are likely to cancel their subscription and help the telecom company take proactive retention actions.**

### Why Customer Churn Matters

Retaining an existing customer can be more cost-effective than acquiring a new customer.

By identifying high-risk customers, the company can:

* Target customers with retention campaigns
* Offer personalized discounts or plans
* Improve customer service
* Address service-related issues
* Increase customer lifetime value
* Reduce customer acquisition costs

---

# 📂 Dataset

The project uses the **Telco Customer Churn dataset**.

The dataset contains customer-level information related to:

* Customer demographics
* Account information
* Telecom services
* Contract details
* Payment methods
* Monthly charges
* Total charges
* Customer tenure
* Churn status

### Target Variable

The target variable is:

```text
Churn
```

It contains two possible values:

```text
Yes → Customer churned
No  → Customer did not churn
```

For machine learning, these values are converted into:

```text
0 → No Churn
1 → Churn
```

---

# 🧹 Data Cleaning

The following data-cleaning activities are performed:

1. Load the CSV dataset.
2. Inspect the dataset structure.
3. Check data types.
4. Check missing values.
5. Convert `TotalCharges` from text to numeric.
6. Handle missing `TotalCharges` values.
7. Check for duplicate records.
8. Remove duplicates if present.
9. Remove `customerID` before machine learning because it is an identifier rather than a predictive feature.

---

# 📊 Descriptive Analysis

## What Happened?

Descriptive analysis provides an overview of the telecom customer base.

The analysis examines:

* Customer churn distribution
* Gender
* Senior citizen status
* Partner status
* Dependents
* Internet service
* Contract type
* Payment method
* Customer tenure
* Monthly charges
* Total charges

### Key Questions

* How many customers have churned?
* What percentage of customers have churned?
* What types of contracts do customers have?
* What is the average customer tenure?
* What are the typical monthly charges?
* Which services are commonly used?

---

# 🔎 Diagnostic Analysis

## Why Are Customers Churning?

Diagnostic analysis investigates the factors associated with customer churn.

The project compares churn behavior across:

* Contract type
* Internet service
* Payment method
* Senior citizen status
* Partner status
* Dependents
* Customer tenure
* Monthly charges
* Total charges

### Example Questions

* Does contract type affect churn?
* Do customers with shorter tenure churn more frequently?
* Does monthly billing amount relate to churn?
* Does internet service type affect churn?
* Are certain payment methods associated with higher churn?

The objective is to identify customer segments that require additional retention attention.

---

# 🤖 Predictive Analysis

## Machine Learning Model

A **Logistic Regression** classification model is used to predict customer churn.

Logistic Regression is appropriate because this is a **binary classification problem**.

### Prediction Classes

| Value | Meaning  |
| ----- | -------- |
| `0`   | No Churn |
| `1`   | Churn    |

The model also calculates the **probability of churn** for each customer.

For example:

```text
Customer → Churn Probability = 82%
Risk Category → High Risk
```

This probability-based approach allows the company to prioritize customers rather than treating every customer equally.

---

# ⚙️ Machine Learning Workflow

The predictive analysis follows this workflow:

```text
Raw Customer Data
        ↓
Data Cleaning
        ↓
Feature Selection
        ↓
Train-Test Split
        ↓
Numerical Feature Scaling
        ↓
Categorical Feature Encoding
        ↓
Logistic Regression
        ↓
Churn Prediction
        ↓
Churn Probability
        ↓
Risk Classification
        ↓
Business Retention Strategy
```

---

# 🔧 Feature Preprocessing

## Numerical Features

The following numerical features are standardized using `StandardScaler`:

* `SeniorCitizen`
* `tenure`
* `MonthlyCharges`
* `TotalCharges`

## Categorical Features

Categorical variables are transformed using `OneHotEncoder`.

These include:

* `gender`
* `Partner`
* `Dependents`
* `PhoneService`
* `MultipleLines`
* `InternetService`
* `OnlineSecurity`
* `OnlineBackup`
* `DeviceProtection`
* `TechSupport`
* `StreamingTV`
* `StreamingMovies`
* `Contract`
* `PaperlessBilling`
* `PaymentMethod`

---

# 📈 Model Evaluation

The Logistic Regression model is evaluated using several classification metrics.

## Accuracy

Measures the percentage of total predictions that were correct.

## Precision

Measures how many customers predicted as churners actually churned.

## Recall

Measures how many actual churners were correctly identified.

Recall is particularly important in this business problem because failing to identify a customer who is about to churn can result in the loss of that customer.

## F1 Score

The F1 score provides a balance between precision and recall.

## ROC-AUC

ROC-AUC measures how effectively the model distinguishes between churn and non-churn customers.

---

# 📉 Visualizations

The project includes multiple visualizations to understand customer behavior and model performance.

### Descriptive Visualizations

* Customer Churn Distribution
* Customers by Contract Type
* Customers by Internet Service
* Monthly Charges Distribution
* Customer Tenure Distribution

### Diagnostic Visualizations

* Contract Type vs Churn
* Internet Service vs Churn
* Payment Method vs Churn
* Tenure vs Churn
* Monthly Charges vs Churn
* Total Charges vs Churn

### Predictive Analysis Visualizations

* Correlation Heatmap
* Confusion Matrix
* ROC Curve
* Customer Risk Category Distribution

---

# 🚦 Customer Risk Segmentation

After generating churn probabilities, customers are divided into three risk categories.

| Churn Probability | Risk Category  |
| ----------------: | -------------- |
|             < 40% | 🟢 Low Risk    |
|      40% – 69.99% | 🟡 Medium Risk |
|             ≥ 70% | 🔴 High Risk   |

These thresholds are used as practical segmentation rules for this case study.

### Business Application

Customers classified as **High Risk** can be prioritized for retention campaigns.

Possible actions include:

* Personalized offers
* Contract incentives
* Customer support follow-up
* Service upgrades
* Loyalty programs
* Billing assistance
* Targeted communication

---

# 👥 High-Risk Customer Identification

The project identifies customers with the highest predicted probability of churn.

The final customer-level prediction output includes:

* Customer ID
* Tenure
* Contract type
* Internet service
* Monthly charges
* Total charges
* Payment method
* Actual churn status
* Churn probability
* Risk category

This creates a practical list that can be used by a customer-retention team.

---

# 💾 Output File

The project generates:

```text
telecom_customer_churn_predictions.csv
```

This file contains the predicted churn probability and risk category for customers.

It can be used for further analysis or provided to a business/retention team for customer targeting.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Development Environment

* Jupyter Notebook
* PyCharm / VS Code

### Version Control

* Git
* GitHub

---

# 📁 Project Structure

```text
Telecom-Customer-Churn/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── Telecom_Customer_Churn.ipynb
│
├── telecom_customer_churn_predictions.csv
│
└── README.md
```

> The prediction CSV is generated after running the notebook.

---

# 🚀 How to Run the Project

## 1. Clone the Repository

Clone this repository to your local computer using Git.

```bash
git clone <your-github-repository-url>
```

## 2. Navigate to the Project Folder

```bash
cd Telecom-Customer-Churn
```

## 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 4. Open the Notebook

Open:

```text
Telecom_Customer_Churn.ipynb
```

using Jupyter Notebook, JupyterLab, VS Code, or PyCharm.

## 5. Update the CSV Path

Make sure the CSV file path in the notebook points to:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

For example:

```python
file_path = r"C:\Users\YourName\ProjectFolder\WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

Alternatively, if the CSV is in the same directory as the notebook:

```python
file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

## 6. Run All Cells

Run the notebook from beginning to end.

The notebook will perform:

```text
Data Loading
      ↓
Data Cleaning
      ↓
Descriptive Analysis
      ↓
Diagnostic Analysis
      ↓
Visualization
      ↓
Predictive Modeling
      ↓
Model Evaluation
      ↓
Churn Risk Prediction
      ↓
Business Insights
```

---

# 📌 Key Business Insights

The analysis can help the telecom company understand which customer characteristics are associated with higher churn.

Important areas to investigate include:

### 1. Contract Type

Customers on shorter contracts may require additional retention attention.

### 2. Customer Tenure

Newer customers may represent an important retention segment.

### 3. Monthly Charges

Customers with higher monthly charges may require targeted pricing or service strategies.

### 4. Internet Service

Different internet-service segments can exhibit different churn patterns.

### 5. Payment Method

Payment methods can be investigated for differences in customer churn behavior.

### 6. Risk-Based Retention

Instead of contacting every customer with the same campaign, the company can prioritize customers according to predicted churn probability.

---

# 💡 Business Recommendation

The telecom company should implement a **proactive customer retention strategy** using churn predictions.

A possible strategy is:

```text
Identify High-Risk Customers
            ↓
Understand Why They May Churn
            ↓
Create Personalized Retention Offer
            ↓
Contact Customer
            ↓
Monitor Customer Response
            ↓
Measure Churn Reduction
```

The machine learning model should therefore be treated as a **decision-support tool**, rather than simply a prediction system.

---

# 🎯 Project Outcome

This project demonstrates how customer data can be transformed into actionable business insights using analytics and machine learning.

The complete workflow covers:

* Data cleaning
* Exploratory data analysis
* Descriptive analytics
* Diagnostic analytics
* Data visualization
* Feature preprocessing
* Classification
* Predictive modeling
* Model evaluation
* Customer risk segmentation
* Business recommendations

The final objective is to help the telecom company **identify potential churners early and take preventive retention action**.

---

# 👨‍💻 Author

**Pravin**

AI Engineering Program — Data Analytics & Machine Learning Case Study

---

# ⭐ Project Summary

> **Telecom Customer Churn Prediction using Python, Data Analytics, Visualization, and Logistic Regression.**

**Business Goal:** Identify customers likely to churn and enable proactive customer retention strategies.
