import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# 🔹 Import MongoDB Client
from utils.mongodb_client import MongoDBClient
from dotenv import load_dotenv
from datetime import datetime, timezone

# ===================== SHAP HELPERS =====================

@st.cache_resource
def load_shap_background():
    try:
        return pd.read_csv("artifacts/shap_background.csv")
    except FileNotFoundError:
        st.error("❌ SHAP background file not found: artifacts/shap_background.csv")
        st.stop()

@st.cache_resource
def load_shap_explainer(_model, background):
    final_model = _model
    if hasattr(_model, "named_steps"):
        final_model = _model.named_steps[list(_model.named_steps.keys())[-1]]

    return shap.LinearExplainer(
        final_model,
        background,
        feature_perturbation="interventional"
    )

# ===================== APP SETUP =====================

load_dotenv()
mongo = MongoDBClient()

model = pickle.load(open("artifacts/model.pkl", "rb"))
MODEL_VERSION = "v1.0"

final_columns = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# ===================== UI =====================

st.title("📞 Telecom Customer Churn Prediction App")
st.write("Fill the customer details below to predict whether the customer will churn.")

# ---------------- USER INPUTS ----------------
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (Months)", 0, 100, 1)
phone = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
payment = st.selectbox(
    "Payment Method",
    ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]
)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# ===================== DATA PREP =====================

data = pd.DataFrame(0, index=[0], columns=final_columns)

data.loc[0, 'SeniorCitizen'] = 1 if senior == "Yes" else 0
data.loc[0, 'tenure'] = tenure
data.loc[0, 'MonthlyCharges'] = monthly_charges
data.loc[0, 'TotalCharges'] = total_charges

data.loc[0, 'gender_Male'] = 1 if gender == "Male" else 0
data.loc[0, 'Partner_Yes'] = 1 if partner == "Yes" else 0
data.loc[0, 'Dependents_Yes'] = 1 if dependents == "Yes" else 0
data.loc[0, 'PhoneService_Yes'] = 1 if phone == "Yes" else 0
data.loc[0, 'MultipleLines_No phone service'] = 1 if multiple_lines == "No phone service" else 0
data.loc[0, 'MultipleLines_Yes'] = 1 if multiple_lines == "Yes" else 0
data.loc[0, 'InternetService_Fiber optic'] = 1 if internet == "Fiber optic" else 0
data.loc[0, 'InternetService_No'] = 1 if internet == "No" else 0
data.loc[0, 'OnlineSecurity_No internet service'] = 1 if online_security == "No internet service" else 0
data.loc[0, 'OnlineSecurity_Yes'] = 1 if online_security == "Yes" else 0
data.loc[0, 'OnlineBackup_No internet service'] = 1 if online_backup == "No internet service" else 0
data.loc[0, 'OnlineBackup_Yes'] = 1 if online_backup == "Yes" else 0
data.loc[0, 'DeviceProtection_No internet service'] = 1 if device_protection == "No internet service" else 0
data.loc[0, 'DeviceProtection_Yes'] = 1 if device_protection == "Yes" else 0
data.loc[0, 'TechSupport_No internet service'] = 1 if tech_support == "No internet service" else 0
data.loc[0, 'TechSupport_Yes'] = 1 if tech_support == "Yes" else 0
data.loc[0, 'StreamingTV_No internet service'] = 1 if streaming_tv == "No internet service" else 0
data.loc[0, 'StreamingTV_Yes'] = 1 if streaming_tv == "Yes" else 0
data.loc[0, 'StreamingMovies_No internet service'] = 1 if streaming_movies == "No internet service" else 0
data.loc[0, 'StreamingMovies_Yes'] = 1 if streaming_movies == "Yes" else 0
data.loc[0, 'Contract_One year'] = 1 if contract == "One year" else 0
data.loc[0, 'Contract_Two year'] = 1 if contract == "Two year" else 0
data.loc[0, 'PaperlessBilling_Yes'] = 1 if paperless == "Yes" else 0
data.loc[0, 'PaymentMethod_Credit card (automatic)'] = 1 if payment == "Credit card (automatic)" else 0
data.loc[0, 'PaymentMethod_Electronic check'] = 1 if payment == "Electronic check" else 0
data.loc[0, 'PaymentMethod_Mailed check'] = 1 if payment == "Mailed check" else 0

# ===================== PREDICTION =====================

st.markdown("### 🔮 Prediction Result")

if st.button("Predict Churn"):
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    result = "Churn" if prediction == 1 else "Not Churn"

    if prediction == 1:
        st.error("❌ Customer is likely to churn")
    else:
        st.success("✔ Customer is not likely to churn")

    st.info(f"📊 Churn Probability: {probability:.2%}")

    # ===================== SHAP =====================
    st.subheader("🔍 Why did the model make this prediction?")

    try:
        background = load_shap_background()[final_columns]
        explainer = load_shap_explainer(model, background)
        shap_values = explainer(data[final_columns])

        shap_df = (
            pd.DataFrame({
                "Feature": final_columns,
                "Impact": shap_values.values[0]
            })
            .assign(AbsImpact=lambda df: df["Impact"].abs())
            .sort_values("AbsImpact", ascending=False)
        )

        top_features = shap_df.head(5)

        st.dataframe(top_features[["Feature", "Impact"]], use_container_width=True)

        st.markdown("### 🧠 Model Interpretation (Plain English)")
        for _, row in top_features.iterrows():
            direction = "increases" if row["Impact"] > 0 else "decreases"
            st.write(f"• **{row['Feature']}** {direction} the likelihood of churn")

        # SHAP bar plot (future-safe)
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], max_display=5, show=False)
        st.pyplot(fig)
        plt.close(fig)


    except Exception as e:
        st.warning(f"⚠️ SHAP explanation could not be generated: {e}")

    # ===================== MONGODB =====================
    record = data.to_dict(orient="records")[0]
    record.update({
        "prediction": result,
        "probability": float(probability),
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone.utc)
    })

    inserted_id = mongo.insert_record(record)

    if inserted_id:
        st.success(f"📁 Record saved to MongoDB (ID: {inserted_id})")
    else:
        st.error("⚠️ Failed to save record in MongoDB")
