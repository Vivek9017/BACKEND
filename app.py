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
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib

# ===========================
# Load ML Objects BEFORE Worker Fork
# ===========================
print("Loading model...")

try:
    model = tf.keras.models.load_model(
        "final_advanced_multi_domain_model.keras"
    )
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    label_encoders = joblib.load("multi_label_encoders.pkl")

    print("Model loaded successfully!")

except Exception as e:
    print("MODEL LOADING ERROR:", str(e))
    raise e

# ===========================
# FastAPI Setup
# ===========================
app = FastAPI(title="Preventive Health AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Preventive Health API Running"}

# ===========================
# Input Schema
# ===========================
class HealthInput(BaseModel):
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    systolic_bp: int
    diastolic_bp: int
    heart_rate: int
    blood_sugar: int
    cholesterol: int
    spo2: int
    body_temperature: float
    smoking: str
    alcohol: str
    exercise_level: str
    diet_type: str
    sleep_hours: float
    stress_level: str
    screen_time_hours: float
    water_intake_l: float

    fatigue: int
    mild_headache: int
    occasional_chest_discomfort: int
    frequent_urination: int
    mild_breathlessness: int
    dry_cough: int
    weight_gain: int
    weight_loss: int
    blurred_vision: int
    dizziness: int
    sleep_disturbance: int
    irregular_heartbeat: int
    leg_swelling: int
    loss_of_appetite: int

# ===========================
# Recommendation Logic
# ===========================
def generate_recommendations(risks):
    recs = []

    if risks["heart_risk"] == "High":
        recs += [
            "Consult cardiologist",
            "Reduce salt intake",
            "Monitor blood pressure weekly"
        ]

    if risks["metabolic_risk"] == "High":
        recs += [
            "Reduce sugar intake",
            "Increase physical activity"
        ]

    if risks["stress_risk"] == "High":
        recs += [
            "Improve sleep schedule",
            "Practice stress management"
        ]

    if risks["lung_risk"] == "High":
        recs += [
            "Avoid smoking",
            "Seek medical advice if breathlessness continues"
        ]

    if risks["lifestyle_risk"] == "High":
        recs += [
            "Start structured exercise routine",
            "Improve diet quality"
        ]

    if not recs:
        recs.append("Maintain your current healthy lifestyle.")

    return recs

# ===========================
# Prediction Endpoint
# ===========================
@app.post("/predict")
def predict(data: HealthInput):

    df = pd.DataFrame([data.dict()])

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

    result = {}

    for i, risk in enumerate(risk_names):
        pred_class = np.argmax(predictions[i], axis=1)
        label = label_encoders[risk].inverse_transform(pred_class)[0]
        confidence = float(np.max(predictions[i]))

        result[risk] = label
        result[f"{risk}_confidence"] = round(confidence, 3)

    recommendations = generate_recommendations(result)

    return {
        "risks": result,
        "recommendations": recommendations
    }
