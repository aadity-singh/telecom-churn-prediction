import pandas as pd
import pickle
import os

# Load trained model (pipeline or estimator)
model = pickle.load(open("artifacts/model.pkl", "rb"))

# Load raw dataset
df = pd.read_csv("Dataset/customer_churn_raw.csv")

# Drop target
df = df.drop(columns=["Churn"])

# One-hot encode same way as training
df_encoded = pd.get_dummies(df, drop_first=True)

# Align columns with model features
final_columns = model.feature_names_in_
df_encoded = df_encoded.reindex(columns=final_columns, fill_value=0)

# Sample small background (important for speed)
background = df_encoded.sample(n=50, random_state=42)

# Save for SHAP
os.makedirs("artifacts", exist_ok=True)
background.to_csv("artifacts/shap_background.csv", index=False)

print("✅ SHAP background created at artifacts/shap_background.csv")
