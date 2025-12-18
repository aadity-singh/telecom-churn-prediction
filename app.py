import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 🔹 Import MongoDB Client
from utils.mongodb_client import MongoDBClient

from dotenv import load_dotenv
import os
from datetime import datetime, timezone
datetime.now(timezone.utc)

load_dotenv()


# Initialize MongoDB connection
mongo = MongoDBClient()

# Load model
model = pickle.load(open("artifacts/model.pkl", "rb"))
MODEL_VERSION = "v1.0"

# Final column order (same as training)
final_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
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
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

st.title("📞 Telecom Customer Churn Prediction App")
st.write("Fill the customer details below to predict whether the customer will churn or not.")

# --------------------- USER INPUT FIELDS ---------------------
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=1)
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
payment = st.selectbox("Payment Method", [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

# --------------- CREATE INPUT DATA FRAME ---------------------
data = pd.DataFrame(columns=final_columns)

# Numeric fields
data.at[0, 'SeniorCitizen'] = 1 if senior == "Yes" else 0
data.at[0, 'tenure'] = tenure
data.at[0, 'MonthlyCharges'] = monthly_charges
data.at[0, 'TotalCharges'] = total_charges

# One-hot encoded categorical fields
data.at[0, 'gender_Male'] = 1 if gender == "Male" else 0
data.at[0, 'Partner_Yes'] = 1 if partner == "Yes" else 0
data.at[0, 'Dependents_Yes'] = 1 if dependents == "Yes" else 0
data.at[0, 'PhoneService_Yes'] = 1 if phone == "Yes" else 0

data.at[0, 'MultipleLines_No phone service'] = 1 if multiple_lines == "No phone service" else 0
data.at[0, 'MultipleLines_Yes'] = 1 if multiple_lines == "Yes" else 0

data.at[0, 'InternetService_Fiber optic'] = 1 if internet == "Fiber optic" else 0
data.at[0, 'InternetService_No'] = 1 if internet == "No" else 0

data.at[0, 'OnlineSecurity_No internet service'] = 1 if online_security == "No internet service" else 0
data.at[0, 'OnlineSecurity_Yes'] = 1 if online_security == "Yes" else 0

data.at[0, 'OnlineBackup_No internet service'] = 1 if online_backup == "No internet service" else 0
data.at[0, 'OnlineBackup_Yes'] = 1 if online_backup == "Yes" else 0

data.at[0, 'DeviceProtection_No internet service'] = 1 if device_protection == "No internet service" else 0
data.at[0, 'DeviceProtection_Yes'] = 1 if device_protection == "Yes" else 0

data.at[0, 'TechSupport_No internet service'] = 1 if tech_support == "No internet service" else 0
data.at[0, 'TechSupport_Yes'] = 1 if tech_support == "Yes" else 0

data.at[0, 'StreamingTV_No internet service'] = 1 if streaming_tv == "No internet service" else 0
data.at[0, 'StreamingTV_Yes'] = 1 if streaming_tv == "Yes" else 0

data.at[0, 'StreamingMovies_No internet service'] = 1 if streaming_movies == "No internet service" else 0
data.at[0, 'StreamingMovies_Yes'] = 1 if streaming_movies == "Yes" else 0

data.at[0, 'Contract_One year'] = 1 if contract == "One year" else 0
data.at[0, 'Contract_Two year'] = 1 if contract == "Two year" else 0
data.at[0, 'PaperlessBilling_Yes'] = 1 if paperless == "Yes" else 0

data.at[0, 'PaymentMethod_Credit card (automatic)'] = 1 if payment == "Credit card (automatic)" else 0
data.at[0, 'PaymentMethod_Electronic check'] = 1 if payment == "Electronic check" else 0
data.at[0, 'PaymentMethod_Mailed check'] = 1 if payment == "Mailed check" else 0

# --------------------- PREDICTION + MONGODB SAVE ---------------------
if st.button("Predict Churn"):

    # Predict
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    result = "Churn" if prediction == 1 else "Not Churn"
    

    # Display result
    if prediction == 1:
        st.error("❌ Customer is likely to churn")
    else:
        st.success("✔ Customer is not likely to churn")
    
    st.info(f"📊 Churn Probability: {probability:.2%}")

    # Prepare record for MongoDB
    record = data.to_dict(orient="records")[0]
    record["prediction"] = result
    record["probability"] = float(probability)
    record["model_version"] = MODEL_VERSION
    record["timestamp"] = datetime.utcnow()
    # Insert into DB
    inserted_id = mongo.insert_record(record)

    if inserted_id:
        st.success(f"📁 Record saved to MongoDB (ID: {inserted_id})")
    else:
        st.error("⚠️ Failed to save record in MongoDB")