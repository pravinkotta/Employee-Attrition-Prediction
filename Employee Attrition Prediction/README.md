# Employee Attrition Prediction — HR Tech Case Study

## 1. Business Problem

Employee turnover is expensive because organizations lose trained people, domain knowledge, productivity, and recruitment costs.

This project builds an **Employee Attrition Prediction** system that estimates which employees may be at higher risk of leaving.

### Business objective

> Predict employees at risk of attrition using role, tenure, satisfaction, compensation, workload, engagement, burnout, and related HR factors so that HR can start early retention conversations.

The prediction should be treated as an **early-warning signal**, not as a final HR decision.

---

## 2. Dataset

**Dataset file:** `employee_attrition_hr_2026.csv`

The dataset contains **5,000 employee records and 29 columns**.

### Target variable

- `attrition`
  - `Yes` = employee left
  - `No` = employee stayed

In the supplied dataset:

- Stayed: 4,105
- Left: 895
- Overall attrition rate: approximately 17.9%

### Main feature groups

#### Employee information
- `age`
- `gender`
- `marital_status`
- `education_level`

#### Job information
- `department`
- `job_role`
- `job_level`

#### Compensation
- `monthly_income`
- `stock_option_level`
- `salary_hike_pct`

#### Career/tenure
- `years_at_company`
- `total_working_years`
- `years_since_promotion`
- `num_prior_companies`

#### Work conditions
- `work_mode`
- `commute_minutes`
- `overtime_hours_per_week`
- `business_travel_days_per_year`
- `training_hours`

#### Employee experience
- `manager_support_score`
- `burnout_score`
- `engagement_score`
- `work_life_balance_score`
- `performance_rating`
- `last_review_score`

#### AI/workplace factors
- `uses_ai_tools_at_work`
- `perceived_ai_job_risk`

---

# 3. Project Structure

```text
Employee_Attrition_HR/
│
├── employee_attrition_hr_2026.csv
├── employee_attrition_prediction.py
├── employee_attrition_risk_predictions.csv
└── README.md
```

---

# 4. Analysis Performed

The project is divided into three major analytics stages:

```text
                 EMPLOYEE ATTRITION
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    DESCRIPTIVE      DIAGNOSTIC     PREDICTIVE
       WHAT?            WHY?          WHAT NEXT?
          │              │              │
          ▼              ▼              ▼
    Understand       Identify       Predict risk
    employee         possible       of leaving
    population       drivers
```

---

# 5. Step 1 — Import Libraries

The project uses:

- pandas → data manipulation
- numpy → numerical operations
- matplotlib → visualization
- seaborn → statistical visualization
- scikit-learn → preprocessing, modeling, and evaluation

---

# 6. Step 2 — Load the Dataset

The Python program automatically looks for:

```text
employee_attrition_hr_2026.csv
```

in the same folder as the Python script.

This makes the project easy to run after cloning the GitHub repository.

---

# 7. Step 3 — Understand the Dataset

The code checks:

- Dataset shape
- First few records
- Data types
- Missing values
- Duplicate rows
- Target distribution

This is important before performing any analysis.

---

# 8. Step 4 — Data Cleaning

The project:

1. Removes exact duplicate rows if present.
2. Excludes `employee_id` from machine-learning features.
3. Converts the target into a binary variable:

```text
No  → 0
Yes → 1
```

`employee_id` is excluded because it identifies an employee but does not represent a meaningful business relationship with attrition.

---

# 9. Step 5 — Descriptive Analysis

Descriptive analysis answers:

> **What is happening in the employee dataset?**

The code calculates:

- Count
- Mean
- Standard deviation
- Minimum
- Maximum
- Quartiles

for numerical variables.

It also examines the distribution of categorical variables such as:

- Department
- Job role
- Work mode
- Education
- Gender

---

# 10. Descriptive Visualizations

The project generates:

### 10.1 Attrition distribution

Shows how many employees stayed versus left.

### 10.2 Attrition rate by department

Helps compare employee turnover across departments.

### 10.3 Attrition rate by job level

Shows whether attrition differs across organizational levels.

### 10.4 Monthly income vs attrition

A boxplot compares compensation distributions for employees who stayed and left.

---

# 11. Step 6 — Diagnostic Analysis

Diagnostic analysis answers:

> **Why might employees be leaving?**

The code compares attrition across:

- Department
- Job role
- Work mode
- Age
- Income
- Tenure
- Promotion delay
- Commute time
- Overtime
- Training
- Manager support
- Burnout
- Engagement
- Work-life balance
- Review score
- Perceived AI job risk

The analysis uses group-level comparisons and correlations to identify possible relationships.

---

# 12. Diagnostic Visualizations

### Correlation matrix

Shows relationships between numerical variables and the attrition target.

### Engagement vs burnout

A scatter plot helps examine whether employees with different engagement and burnout levels have different attrition outcomes.

### Overtime vs attrition

Compares weekly overtime for employees who stayed versus left.

### Work experience indicators

Compares:

- Manager support
- Engagement
- Work-life balance
- Burnout

between employees who stayed and left.

---

# 13. Step 7 — Predictive Analysis

Predictive analysis answers:

> **Which employees may be at higher risk of leaving?**

This project uses:

## Logistic Regression

Logistic Regression is suitable because attrition is a binary classification problem:

```text
0 = Stayed
1 = Left
```

The model produces a probability of attrition.

For example:

```text
Employee A → 0.12 → Low Risk
Employee B → 0.48 → Medium Risk
Employee C → 0.81 → High Risk
```

---

# 14. Data Preprocessing

### Numerical variables

The pipeline:

1. Fills missing values using the median.
2. Standardizes numerical variables.

### Categorical variables

The pipeline:

1. Fills missing categories using the most frequent value.
2. Applies One-Hot Encoding.

Using a pipeline keeps preprocessing and modeling together and helps prevent data leakage.

---

# 15. Train/Test Split

The dataset is divided into:

```text
80% → Training
20% → Testing
```

`stratify=y` is used so the proportion of employees who left and stayed remains similar in both sets.

---

# 16. Model Evaluation

The model reports:

### Accuracy

Overall percentage of correct predictions.

### Precision

Among employees predicted to leave, how many actually left?

### Recall

Among employees who actually left, how many did the model identify?

### F1 Score

Balances precision and recall.

### ROC-AUC

Measures how well the model separates employees who left from employees who stayed across classification thresholds.

For an HR retention use case, **recall is particularly important** because missing a genuinely at-risk employee can mean losing an opportunity for early intervention.

---

# 17. Confusion Matrix

The confusion matrix contains:

```text
                     Predicted
                  Stayed     Left

Actual Stayed       TN        FP

Actual Left         FN        TP
```

The most important error for an early-warning retention system is often:

```text
FN = Actual Left but Predicted Stayed
```

because the system failed to flag an employee who subsequently left.

---

# 18. ROC Curve

The ROC curve shows the trade-off between:

- True Positive Rate
- False Positive Rate

The area under the curve is reported as ROC-AUC.

---

# 19. Model Signals

For Logistic Regression, model coefficients are used to identify the strongest predictive signals.

A positive coefficient pushes the prediction toward:

```text
Higher probability of attrition
```

A negative coefficient pushes the prediction toward:

```text
Lower probability of attrition
```

The coefficient should be interpreted as a model association, **not proof of causation**.

---

# 20. Employee Risk Scoring

The model calculates:

```text
attrition_probability
```

for every employee.

Employees are grouped into:

```text
0% – 30%    → Low Risk
30% – 60%   → Medium Risk
60% – 100%  → High Risk
```

These thresholds are business-oriented starting points and can be changed after HR validates the operating requirements.

---

# 21. Output File

The program creates:

```text
employee_attrition_risk_predictions.csv
```

The output contains:

- Employee ID
- Department
- Job role
- Job level
- Years at company
- Monthly income
- Engagement score
- Burnout score
- Manager support score
- Attrition probability
- Risk band

This file can be used for further HR analysis and dashboarding.

---

# 22. Business Use Case

A possible HR workflow is:

```text
Model predicts high risk
          ↓
HR reviews employee context
          ↓
Retention conversation
          ↓
Understand underlying concerns
          ↓
Possible intervention
          ↓
Monitor employee experience
          ↓
Re-score periodically
```

Possible areas for discussion include:

- Workload
- Burnout
- Manager support
- Compensation
- Career growth
- Promotion opportunities
- Work-life balance
- Engagement
- Training needs

---

# 23. Important HR / AI Governance Note

The model should **not** be used to automatically:

- terminate an employee
- deny a promotion
- reduce compensation
- label an employee as disloyal
- make a disciplinary decision

A prediction indicates risk, not certainty.

HR should combine the model with:

- Human review
- Fairness checks
- Appropriate access controls
- Explainability
- Employee privacy
- Regular model monitoring

Sensitive demographic attributes should be handled carefully and reviewed for potential fairness issues.

---

# 24. Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# 25. How to Run

Clone the repository and place the CSV in the same folder as the Python file.

Run:

```bash
python employee_attrition_prediction.py
```

The script will:

1. Load the dataset
2. Clean the data
3. Perform descriptive analysis
4. Generate descriptive visualizations
5. Perform diagnostic analysis
6. Generate diagnostic visualizations
7. Train the Logistic Regression model
8. Evaluate the model
9. Generate predictive visualizations
10. Calculate employee attrition probabilities
11. Create the risk prediction CSV

---

# 26. Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Logistic Regression
- One-Hot Encoding
- StandardScaler
- Classification metrics

---

# 27. Key Learning Outcomes

This project demonstrates how to build an end-to-end HR analytics classification workflow:

```text
Business Problem
       ↓
Data Loading
       ↓
Data Understanding
       ↓
Data Cleaning
       ↓
Descriptive Analytics
       ↓
Visualization
       ↓
Diagnostic Analytics
       ↓
Feature Preparation
       ↓
Classification Model
       ↓
Model Evaluation
       ↓
Employee Risk Scoring
       ↓
Business Recommendation
```

This makes the project suitable as an **AI/ML portfolio case study** because it connects technical modeling with a real business problem.

---

# 28. Future Improvements

Possible next steps:

1. Compare Logistic Regression with Random Forest.
2. Try Gradient Boosting / XGBoost.
3. Tune the probability threshold based on HR capacity.
4. Perform cross-validation.
5. Add fairness analysis across demographic groups.
6. Use SHAP for explainability.
7. Build a Power BI or Streamlit dashboard.
8. Add model monitoring.
9. Test the model on future employee data.
10. Create a retention recommendation engine.

---

## Conclusion

This project demonstrates how HR data can be transformed into an early-warning attrition system.

The most important business idea is:

> **Use machine learning to identify where HR should look, then use human judgment to understand why and decide what action is appropriate.**
