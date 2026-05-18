# Customer Churn Prediction

An end-to-end machine learning system that predicts whether a telecom customer will cancel their subscription — from raw data to a deployed REST API.

---

## The Business Problem

Acquiring a new customer costs 5-7x more than retaining an existing one. If a company can identify customers likely to leave before they do, the retention team can intervene with targeted offers. This project builds a model that flags high-risk customers and exposes it through an API that any application can call.

---

## Project Structure

```
customer-churn/
├── Customer_Churn.ipynb       # Full analysis notebook
├── app.py                     # FastAPI prediction endpoint
├── churn_model.joblib         # Trained Gradient Boosting model
├── churn_scaler.joblib        # Fitted StandardScaler
├── churn_columns.joblib       # Training column names and order
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

---

## Dataset

- **Source:** IBM Telco Customer Churn (Kaggle)
- **Rows:** 7,043 customers
- **Features:** 20 features — demographics, services, contract details
- **Target:** Churn — Yes/No (26.5% churn rate)

---

## Full Pipeline

### 1. Data Cleaning
- Discovered `TotalCharges` stored as object dtype due to 11 blank strings
- Converted to numeric with `pd.to_numeric(errors='coerce')`
- Dropped 11 rows with zero tenure — brand new customers with no churn signal
- Converted target from Yes/No strings to binary 0/1

### 2. Exploratory Data Analysis
Key findings before any modeling:

| Feature | Finding |
|---|---|
| Tenure | Churned customers leave early — mean 18 months vs 38 for retained |
| MonthlyCharges | Churned customers pay more — $74/month vs $61 |
| Contract | Month-to-month: 43% churn vs 2.5% for two-year contracts |
| InternetService | Fiber optic customers churn at 42% |
| PaymentMethod | Electronic check customers churn at 45% |
| OnlineSecurity | No security: 42% churn vs 15% with security |

### 3. Feature Engineering
- Dropped `TotalCharges` — 0.83 correlation with tenure (multicollinearity)
- One-hot encoded 15 categorical features with `drop_first=True`
- Created `num_services` — count of active services per customer. Retained customers average 3.0 services vs 2.7 for churners

### 4. Class Imbalance
26.5% churn rate — moderate imbalance handled with `class_weight='balanced'` passed directly to the model. More aggressive techniques like SMOTE were not needed at this imbalance level.

### 5. Model Comparison (5-fold cross validation, ROC-AUC)

| Model | ROC-AUC | Std |
|---|---|---|
| Logistic Regression | 0.849 | 0.020 |
| Gradient Boosting | 0.847 | 0.016 |
| Random Forest | 0.825 | 0.016 |

Gradient Boosting selected for its tuning potential and industry relevance. The near-identical score of Logistic Regression suggests churn patterns in this dataset are largely linear.

### 6. Hyperparameter Tuning
GridSearchCV with 5-fold CV across 36 combinations:
```
Best params: learning_rate=0.05, max_depth=3, n_estimators=100, subsample=1.0
Best CV ROC-AUC: 0.848
```
Marginal improvement confirms the model was already near its performance ceiling — further gains require richer features, not better hyperparameters.

### 7. Final Evaluation

**ROC-AUC: 0.842** on held-out test set

| Threshold | Churn Recall | Churn Precision | Churners Missed |
|---|---|---|---|
| 0.5 (default) | 51% | 65% | 182 |
| 0.3 (tuned) | 79% | 53% | 77 |

At threshold 0.3 the model catches 4 in 5 churners before they leave. The increase in false positives is an acceptable business tradeoff — a retention call costs far less than losing a customer.

### 8. Feature Importance

Top drivers of churn identified by the model:

1. **Tenure** — by far the strongest signal. Short tenure = high risk
2. **Fiber optic internet** — possibly overpriced or underdelivering
3. **Electronic check payment** — correlates with less engaged customers
4. **Contract type** — month-to-month customers have no switching cost
5. **MonthlyCharges** — higher bills increase churn probability

These match exactly what was identified manually during EDA — before any model was built.

---

## REST API

Built with FastAPI. Accepts raw customer data, handles preprocessing internally, and returns churn probability with a binary prediction.

### Run the API

```bash
pip install fastapi uvicorn joblib scikit-learn pandas
python -m uvicorn app:app --reload
```

Interactive docs available at: `http://127.0.0.1:8000/docs`

### Endpoints

**GET /** — Health check
```json
{"message": "Customer Churn Prediction API is running"}
```

**POST /predict** — Predict churn for a customer

Example request:
```json
{
  "SeniorCitizen": 0,
  "tenure": 2,
  "MonthlyCharges": 85.0,
  "gender": "Female",
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}
```

Example response:
```json
{
  "churn_probability": 0.8923,
  "will_churn": true,
  "threshold_used": 0.3
}
```

### Valid Field Values

| Field | Valid Values |
|---|---|
| gender | Male, Female |
| Partner, Dependents, PaperlessBilling | Yes, No |
| PhoneService | Yes, No |
| MultipleLines | Yes, No, No phone service |
| InternetService | DSL, Fiber optic, No |
| OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | Yes, No, No internet service |
| Contract | Month-to-month, One year, Two year |
| PaymentMethod | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |

---

## Results Summary

- **ROC-AUC:** 0.842 — model correctly ranks a churner above a non-churner 84.2% of the time
- **Recall at 0.3 threshold:** 79% of churners identified before they leave
- **Deployed:** Working REST API with automatic input validation and preprocessing

---

## Limitations

The dataset lacks behavioral signals that would be the strongest real-world churn indicators — customer service call history, complaint records, network usage patterns, and satisfaction scores. The 0.842 ROC-AUC reflects the ceiling of what these features can predict, not the ceiling of what's possible with richer data.

---

## If I Had More Time

- Test Logistic Regression as final model — scored nearly identically and fully interpretable via coefficients
- Try XGBoost with `scale_pos_weight` on a larger dataset
- Collect behavioral features — usage patterns, support tickets, satisfaction scores
- Add a Pipeline to handle preprocessing inside GridSearchCV properly
- Deploy to a cloud service (Railway, Render, or AWS)

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | Modeling and evaluation |
| Matplotlib, Seaborn | Visualization |
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| Joblib | Model serialization |


## Live Demo
API is deployed and accessible at: https://customer-churn-prediction-production-25ab.up.railway.app/docs

