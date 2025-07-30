
import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load Model (with caching)
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load('diabetic_retinopathy_logistic_model.pkl')

model = load_model()

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="centered")

# -------------------------
# Custom CSS Styling (Improvement 1)
# -------------------------
st.markdown("""
    <style>
        .main {
            background-color: #fefefe;
        }
        h1 {
            font-family: 'Helvetica Neue', sans-serif;
            color: #2c3e50;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# App Title & Description
# -------------------------
st.title("👁️🩺 Diabetic Retinopathy Prediction App")
st.markdown("This app predicts whether a person shows signs of diabetic retinopathy based on input health features.")

# -------------------------
# Lottie Animation (Improvement 2)
# -------------------------
from streamlit_lottie import st_lottie
import requests

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a medical animation
lottie_medical = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_tutvdkg0.json")

# Display the animation
if lottie_medical:
    st_lottie(lottie_medical, height=200)
else:
    st.warning("⚠️ Animation couldn't load.")


# -------------------------
# Sidebar Info
# -------------------------
with st.sidebar.expander("ℹ️ About this app"):
    st.markdown("""
    - ✅ **Purpose**: Predict Diabetic Retinopathy presence
    - ⚙️ **Model**: Logistic Regression (scikit-learn)
    - 📐 **Derived Features**: Pulse Pressure & MAP
    - 🧠 **Built with**: Streamlit + Joblib
    """)

# -------------------------
# Input Form
# Form-based input layout
# -------------------------
with st.form("input_form"):
    st.subheader("📝 Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=30, max_value=105, value=50)
        systolic_bp = st.number_input('Systolic Blood Pressure', min_value=70.0, max_value=130.0, value=120.0)

    with col2:
        cholesterol = st.number_input('Cholesterol Level', min_value=70.0, max_value=130.0, value=90.0)
        diastolic_bp = st.number_input('Diastolic Blood Pressure', min_value=60.0, max_value=120.0, value=80.0)

    submitted = st.form_submit_button("🔍 Predict")

# -------------------------
# Feature Engineering
# -------------------------
pulse_pressure = systolic_bp - diastolic_bp
mean_arterial_pressure = (systolic_bp + 2 * diastolic_bp) / 3

# Prepare input DataFrame
input_df = pd.DataFrame([{
    'age': age,
    'systolic_bp': systolic_bp,
    'diastolic_bp': diastolic_bp,
    'cholesterol': cholesterol,
    'pulse_pressure': pulse_pressure,
    'mean_arterial_pressure': mean_arterial_pressure
}])

# -------------------------
# Prediction
# -------------------------
if submitted:
    # Make prediction
    prediction = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    confidence = pred_proba[prediction]

    # Derived Features Box
    # Display derived features (optional)
    # Detect Streamlit theme (light or dark)
    theme = st.get_option("theme.base")
    is_dark = theme == "dark"

    # Set background color accordingly
    bg_color = "#2b2b2b" if is_dark else "#f0f0f5"
    text_color = "#ffffff" if is_dark else "#000000"

    # Render the styled box
    st.markdown(f"""
        <div style='
            padding: 10px;
            background-color: {bg_color};
            color: {text_color};
            border-radius: 10px;
            margin-top: 10px;
        '>
            <b>Pulse Pressure:</b> {pulse_pressure:.2f} mmHg<br>
            <b>Mean Arterial Pressure:</b> {mean_arterial_pressure:.2f} mmHg
        </div>
    """, unsafe_allow_html=True)


    # Display prediction result
    st.markdown("### 🔍 Prediction Result")
    if prediction == 1:
        st.error(f"🧪 The model predicts **presence** of Diabetic Retinopathy (Confidence: {confidence:.2f})")
    else:
        st.success(f"✅ The model predicts **no signs** of Diabetic Retinopathy (Confidence: {confidence:.2f})")

    # Confidence Progress, Add a Progress Bar for Confidence
    st.write("📊 Model Confidence:")
    st.progress(confidence)


  

    # Download Prediction Report
    report = f"""
Prediction: {"DR Present" if prediction else "No DR"}
Confidence: {confidence:.2f}
Pulse Pressure: {pulse_pressure:.2f}
Mean Arterial Pressure: {mean_arterial_pressure:.2f}
"""
    st.download_button("📄 Download Report", report, file_name="dr_prediction_report.txt")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("👨‍⚕️ Created with ❤️ by Sathwik Patel using Streamlit")
