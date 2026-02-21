# ===========================
# TensorFlow Memory Optimization
# ===========================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ===========================
# Other Imports
# ===========================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===========================
# Load Model
# ===========================
@st.cache_resource
def load_objects():
    model = tf.keras.models.load_model("final_advanced_multi_domain_model.keras")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    label_encoders = joblib.load("multi_label_encoders.pkl")
    return model, scaler, feature_columns, label_encoders

st.title("🩺 Preventive Health AI")

st.write("Fill in your health details below to predict risk levels.")

model, scaler, feature_columns, label_encoders = load_objects()

# ===========================
# User Inputs
# ===========================
age = st.number_input("Age", 1, 120)
gender = st.selectbox("Gender", ["Male", "Female"])
height_cm = st.number_input("Height (cm)")
weight_kg = st.number_input("Weight (kg)")
systolic_bp = st.number_input("Systolic BP")
diastolic_bp = st.number_input("Diastolic BP")
heart_rate = st.number_input("Heart Rate")
blood_sugar = st.number_input("Blood Sugar")
cholesterol = st.number_input("Cholesterol")
spo2 = st.number_input("SpO2")
body_temperature = st.number_input("Body Temperature")
smoking = st.selectbox("Smoking", ["Yes", "No"])
alcohol = st.selectbox("Alcohol", ["Yes", "No"])
exercise_level = st.selectbox("Exercise Level", ["Low", "Moderate", "High"])
diet_type = st.selectbox("Diet Type", ["Veg", "Non-Veg", "Mixed"])
sleep_hours = st.number_input("Sleep Hours")
stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
screen_time_hours = st.number_input("Screen Time (hours)")
water_intake_l = st.number_input("Water Intake (liters)")

fatigue = st.checkbox("Fatigue")
mild_headache = st.checkbox("Mild Headache")
occasional_chest_discomfort = st.checkbox("Chest Discomfort")
frequent_urination = st.checkbox("Frequent Urination")
mild_breathlessness = st.checkbox("Breathlessness")
dry_cough = st.checkbox("Dry Cough")
weight_gain = st.checkbox("Weight Gain")
weight_loss = st.checkbox("Weight Loss")
blurred_vision = st.checkbox("Blurred Vision")
dizziness = st.checkbox("Dizziness")
sleep_disturbance = st.checkbox("Sleep Disturbance")
irregular_heartbeat = st.checkbox("Irregular Heartbeat")
leg_swelling = st.checkbox("Leg Swelling")
loss_of_appetite = st.checkbox("Loss of Appetite")

# ===========================
# Predict Button
# ===========================
if st.button("Predict Risk"):

    input_dict = {
        "age": age,
        "gender": gender,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
        "blood_sugar": blood_sugar,
        "cholesterol": cholesterol,
        "spo2": spo2,
        "body_temperature": body_temperature,
        "smoking": smoking,
        "alcohol": alcohol,
        "exercise_level": exercise_level,
        "diet_type": diet_type,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "screen_time_hours": screen_time_hours,
        "water_intake_l": water_intake_l,
        "fatigue": int(fatigue),
        "mild_headache": int(mild_headache),
        "occasional_chest_discomfort": int(occasional_chest_discomfort),
        "frequent_urination": int(frequent_urination),
        "mild_breathlessness": int(mild_breathlessness),
        "dry_cough": int(dry_cough),
        "weight_gain": int(weight_gain),
        "weight_loss": int(weight_loss),
        "blurred_vision": int(blurred_vision),
        "dizziness": int(dizziness),
        "sleep_disturbance": int(sleep_disturbance),
        "irregular_heartbeat": int(irregular_heartbeat),
        "leg_swelling": int(leg_swelling),
        "loss_of_appetite": int(loss_of_appetite),
    }

    df = pd.DataFrame([input_dict])
    df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)

    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)

    df_scaled = scaler.transform(df)
    predictions = model.predict(df_scaled)

    risk_names = [
        "heart_risk",
        "metabolic_risk",
        "stress_risk",
        "lung_risk",
        "lifestyle_risk"
    ]

    st.subheader("Prediction Results")

    for i, risk in enumerate(risk_names):
        pred_class = np.argmax(predictions[i], axis=1)
        label = label_encoders[risk].inverse_transform(pred_class)[0]
        confidence = float(np.max(predictions[i]))
        st.write(f"**{risk}**: {label} (Confidence: {round(confidence,3)})")
