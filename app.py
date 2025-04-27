import joblib
import pandas as pd
import streamlit as st

# --- Efficient model and scaler loading ---
@st.cache_resource()
def load_model_scaler():
    rf_model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return rf_model, scaler

rf_model, scaler = load_model_scaler()

# --- Streamlit UI ---
st.title("🩺 Breast Cancer 10-Year Mortality Prediction App")

st.write("This app predicts the **10-year survival** for breast cancer patients based on clinical features.")

# --- User Inputs ---
st.header("Enter Patient Details:")

age = st.number_input("Age at Diagnosis", 20, 100)
tumor_size = st.number_input("Tumor Size (in mm)", 0.0, 200.0)
grade = st.selectbox("Neoplasm Histologic Grade", [1, 2, 3])
tumor_stage = st.selectbox("Tumor Stage", ['Stage I', 'Stage II', 'Stage III', 'Stage IV'])
er_status = st.selectbox("ER Status", ["Positive", "Negative"])
pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
her2_status = st.selectbox("HER2 Status", ["Positive", "Negative"])
hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])
radio_therapy = st.selectbox("Radio Therapy", ["Yes", "No"])

# --- Mapping tumor stage ---
tumor_stage_mapping = {'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IV': 4}
tumor_stage_numeric = tumor_stage_mapping[tumor_stage]

# --- Prepare input data ---
input_data = {
    'Age at Diagnosis': age,
    'Tumor Size': tumor_size,
    'Neoplasm Histologic Grade': grade,
    'Tumor Stage': tumor_stage_numeric,
    'ER Status_Positive': 1 if er_status == "Positive" else 0,
    'PR Status_Positive': 1 if pr_status == "Positive" else 0,
    'HER2 Status_Positive': 1 if her2_status == "Positive" else 0,
    'Hormone Therapy_Yes': 1 if hormone_therapy == "Yes" else 0,
    'Radio Therapy_Yes': 1 if radio_therapy == "Yes" else 0
}

input_df = pd.DataFrame([input_data])

# --- Prediction Button ---
if st.button("🔍 Predict 10-Year Survival"):
    # Scale input
    scaled_input = scaler.transform(input_df)
    
    # Predict
    prediction = rf_model.predict(scaled_input)
    
    # Show result
    if prediction[0] == 1:
        st.success("🎯 Patient is Likely to Survive More Than 10 Years")
    else:
        st.error("⚠️ Patient is at Risk of Mortality Within 10 Years")
