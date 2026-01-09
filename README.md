# 📞 Telecom Customer Churn Prediction App

An end-to-end **Machine Learning web application** that predicts whether a telecom customer is likely to churn based on demographic, billing, and service usage details.
The app also provides **model explainability using SHAP**, allowing users to understand *why* a particular prediction was made.
Built with **Streamlit**, trained using **scikit-learn**, and deployed on **Render**.

---

## 🚀 Live Demo
🔗 https://telecom-churn-prediction-ea93.onrender.com

---

## 🧠 Problem Statement
Customer churn is a major challenge in the telecom industry.  
Retaining existing customers is significantly cheaper than acquiring new ones.

This project helps:
- Identify customers at high risk of churn
- Understand key factors influencing churn decisions
- Support data-driven retention strategies

---

## 🔍 Key Features
- 📋 Interactive form for customer details
- 🔮 Real-time churn prediction
- 📊 Churn probability score
- 🧠 Plain-English explanation of prediction
- 📈 SHAP-based feature impact analysis
- 🗄 Prediction records stored in MongoDB
- ☁️ Production-ready deployment

---

## 🛠 Tech Stack
- **Python**
- **Streamlit**
- **Scikit-learn**
- **Pandas & NumPy**
- **SHAP (Explainable AI)**
- **MongoDB**
- **Render (Deployment)**

---

## 📊 Model Details
- Algorithm: Logistic Regression (ML classifier)
- Input: Customer demographic, billing, and service features
- Output:
  - `Churn` / `Not Churn`
  - Churn probability
  - Feature-wise impact explanation

---

## 🧠 Model Explainability (SHAP)
The application uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

For each prediction:
- The top contributing features are identified
- Positive values increase churn likelihood
- Negative values decrease churn likelihood
- Results are shown in both:
  - Tabular format
  - Visual bar chart

This makes the model **transparent and interpretable**, even for non-technical users.

---

## 🗂 Project Structure
Telecom Churn Project/
│
├── app.py
├── requirements.txt
├── artifacts/
│ ├── model.pkl
│ └── shap_background.csv
├── utils/
│ └── mongodb_client.py
├── Dataset/
│ └── customer_churn_raw.csv
└── README.md
