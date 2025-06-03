import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('linear_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🎓 Student Exam Score Predictor")

# 1. Collect user inputs

age = st.slider("Age", 17, 24, 20)
study_hours = st.slider("Study hours per day", 0.0, 10.0, 3.5)
social_hours = st.slider("Social media hours", 0.0, 8.0, 2.5)
netflix_hours = st.slider("Netflix hours", 0.0, 6.0, 1.5)
attendance = st.slider("Attendance %", 50.0, 100.0, 85.0)
sleep_hours = st.slider("Sleep hours", 3.0, 10.0, 6.5)
exercise = st.slider("Exercise frequency (per week)", 0, 7, 3)
mental_health = st.slider("Mental health rating (1-10)", 1, 10, 5)

part_time_job = st.selectbox("Part-time job", ["No", "Yes"])
part_time_job = 1 if part_time_job == "Yes" else 0

extracurricular = st.selectbox("Extracurricular participation", ["No", "Yes"])
extracurricular = 1 if extracurricular == "Yes" else 0

gender = st.selectbox("Gender", ["Female", "Male", "Other"])
gender_male = 1 if gender == "Male" else 0
gender_other = 1 if gender == "Other" else 0

diet_quality = st.selectbox("Diet Quality", ["Poor", "Fair", "Good"])
diet_fair = 1 if diet_quality == "Fair" else 0
diet_good = 1 if diet_quality == "Good" else 0

internet_quality = st.selectbox("Internet Quality", ["Poor", "Average", "Good"])
internet_avg = 1 if internet_quality == "Average" else 0
internet_good = 1 if internet_quality == "Good" else 0

# 2. Prepare feature array (order MUST match training data)

features = np.array([[
    age,
    study_hours,
    social_hours,
    netflix_hours,
    attendance,
    sleep_hours,
    exercise,
    mental_health,
    part_time_job,
    extracurricular,
    gender_male,
    gender_other,
    diet_fair,
    diet_good,
    internet_avg,
    internet_good
]], dtype=float)

# 3. Scale features

features_scaled = scaler.transform(features)

# 4. Predict and display result

if st.button("Predict Exam Score"):
    prediction = model.predict(features_scaled)[0]
    prediction = max(0, min(100, prediction))  # Keep score between 0 and 100
    st.success(f"🎯 Predicted Exam Score: {prediction:.2f}")
