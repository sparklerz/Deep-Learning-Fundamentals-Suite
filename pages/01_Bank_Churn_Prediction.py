import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from src.hf_utils import hf_download, hf_load_pickle, hf_read_json

REPO_ID = "ash001/bank-churn-ann"

st.set_page_config(page_title="Bank Churn Prediction", page_icon="🏦", layout="wide")
st.title("🏦 Bank Customer Churn Prediction (ANN)")
st.caption("Repo: [ash001/bank-churn-ann](https://huggingface.co/ash001/bank-churn-ann)")

schema = hf_read_json(REPO_ID, "artifacts/schema.json")
# Churn datasets are usually imbalanced (~20% churn). A 0.5 threshold often predicts "stay".
threshold_default = 0.35

# Initialize widget state
DEFAULTS = {
    "credit_score": 650,
    "age": 40,
    "tenure": 5,
    "geography": "France",
    "gender": "Female",
    "num_products": 2,
    "balance": 50000.0,
    "has_card": 1,
    "is_active": 1,
    "est_salary": 100000.0,
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

@st.cache_resource(show_spinner=False)
def load_assets():
    model_path = hf_download(REPO_ID, "artifacts/model.h5")
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = hf_load_pickle(REPO_ID, "artifacts/scaler.pkl")
    le_gender = hf_load_pickle(REPO_ID, "artifacts/label_encoder_gender.pkl")
    ohe_geo = hf_load_pickle(REPO_ID, "artifacts/onehot_encoder_geo.pkl")
    return model, scaler, le_gender, ohe_geo

model, scaler, le_gender, ohe_geo = load_assets()

# Optional: load dataset for "sample row" convenience
@st.cache_data(show_spinner=False)
def load_dataset():
    csv_path = hf_download(REPO_ID, "Churn_Modelling.csv")
    df = pd.read_csv(csv_path)
    # match training cleanup
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, errors="ignore")
    return df

with st.expander("Use a random row from the training dataset (optional)", expanded=False):
    df = load_dataset()
    st.caption(f"Dataset rows: **{len(df)}** (valid row index: 0 to {len(df)-1})")
    idx = st.number_input("Row index", min_value=0, max_value=int(len(df) - 1), value=0, step=1)
    row_df = df.iloc[[int(idx)]]
    st.write(row_df)
    row = row_df.iloc[0]

    if st.button("Load this row into the inputs"):
        st.session_state["credit_score"] = int(row["CreditScore"])
        st.session_state["age"] = int(row["Age"])
        st.session_state["tenure"] = int(row["Tenure"])
        st.session_state["geography"] = str(row["Geography"])
        st.session_state["gender"] = str(row["Gender"])
        st.session_state["num_products"] = int(row["NumOfProducts"])
        st.session_state["balance"] = float(row["Balance"])
        st.session_state["has_card"] = int(row["HasCrCard"])
        st.session_state["is_active"] = int(row["IsActiveMember"])
        st.session_state["est_salary"] = float(row["EstimatedSalary"])
        st.success(f"Loaded row {int(idx)} into inputs. True label Exited = {int(row['Exited'])}")

# Inputs
c1, c2, c3 = st.columns(3)
with c1:
    credit_score = st.number_input("CreditScore", min_value=300, max_value=900, step=1, key="credit_score")
    age = st.number_input("Age", min_value=18, max_value=100, step=1, key="age")
    tenure = st.number_input("Tenure", min_value=0, max_value=10, step=1, key="tenure")
with c2:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"], key="geography")
    gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
    num_products = st.number_input("NumOfProducts", min_value=1, max_value=4, step=1, key="num_products")
with c3:
    balance = st.number_input("Balance", min_value=0.0, step=1000.0, format="%.2f", key="balance")
    has_card = st.selectbox("HasCreditCard", [0, 1], key="has_card")
    is_active = st.selectbox("IsActiveMember", [0, 1], key="is_active")
    est_salary = st.number_input("EstimatedSalary", min_value=0.0, step=1000.0, format="%.2f", key="est_salary")

threshold = st.slider("Decision threshold", 0.05, 0.95, float(threshold_default), 0.01)

def build_feature_vector():
    # Encode Gender
    gender_enc = int(le_gender.transform([gender])[0])

    # One-hot Geography
    geo_arr = ohe_geo.transform([[geography]])
    geo_cols = schema["geo_onehot_columns"]
    geo_df = pd.DataFrame(geo_arr, columns=geo_cols)

    # Order features exactly as schema.feature_names
    row = {
        "CreditScore": credit_score,
        "Gender": gender_enc,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": est_salary,
    }
    base_df = pd.DataFrame([row])
    X = pd.concat([base_df, geo_df], axis=1)

    X = X[schema["feature_names"]]
    X_scaled = scaler.transform(X.values.astype(np.float32))
    return X_scaled

if st.button("Predict churn", type="primary"):
    X_scaled = build_feature_vector()
    prob = float(model.predict(X_scaled, verbose=0).reshape(-1)[0])
    pred = int(prob >= threshold)

    st.metric("Churn probability", f"{prob:.3f}")
    st.write("Prediction:", "✅ Likely to exit" if pred == 1 else "🟦 Likely to stay")
    st.caption("Tip: Adjust the threshold based on your desired precision/recall tradeoff.")
