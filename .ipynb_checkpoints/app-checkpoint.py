import streamlit as st
import numpy as np
import pickle

# 🎯 Load model, scaler, and features
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features_list = pickle.load(open("features.pkl", "rb"))

# 🧠 Title
st.title("🧠 Mental Health Risk Prediction 💡")
st.write("Fill the details to predict if treatment is needed")

# 📋 Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
occupation = st.selectbox("Occupation", ["Student", "Working Professional", "Self-employed", "Unemployed"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
family_history = st.selectbox("Family History", ["Yes", "No"])
days_indoors = st.selectbox("Days Indoors", ["1-14", "15-30", "More than 30"])
growing_stress = st.selectbox("Growing Stress", ["Yes", "No"])
changes_habits = st.selectbox("Changes in Habits", ["Yes", "No"])
mental_history = st.selectbox("Mental Health History", ["Yes", "No"])
mood_swings = st.selectbox("Mood Swings", ["Yes", "No"])
coping_struggles = st.selectbox("Coping Struggles", ["Yes", "No"])
work_interest = st.selectbox("Work Interest", ["Low", "Medium", "High"])
social_weakness = st.selectbox("Social Weakness", ["Yes", "No"])
interview = st.selectbox("Mental Health Interview", ["Yes", "No"])
care_options = st.selectbox("Care Options", ["Yes", "No"])

# 🔄 Encoding
gender = 1 if gender == "Male" else 0
self_employed = 1 if self_employed == "Yes" else 0
family_history = 1 if family_history == "Yes" else 0
growing_stress = 1 if growing_stress == "Yes" else 0
changes_habits = 1 if changes_habits == "Yes" else 0
mental_history = 1 if mental_history == "Yes" else 0
mood_swings = 1 if mood_swings == "Yes" else 0
coping_struggles = 1 if coping_struggles == "Yes" else 0
social_weakness = 1 if social_weakness == "Yes" else 0
interview = 1 if interview == "Yes" else 0
care_options = 1 if care_options == "Yes" else 0

occupation_map = {"Student": 0, "Working Professional": 1, "Self-employed": 2, "Unemployed": 3}
days_map = {"1-14": 0, "15-30": 1, "More than 30": 2}
work_map = {"Low": 0, "Medium": 1, "High": 2}

occupation = occupation_map[occupation]
days_indoors = days_map[days_indoors]
work_interest = work_map[work_interest]

# 🔍 Prediction
if st.button("Predict"):
    input_data = np.array([[
        gender,
        occupation,
        self_employed,
        family_history,
        days_indoors,
        growing_stress,
        changes_habits,
        mental_history,
        mood_swings,
        coping_struggles,
        work_interest,
        social_weakness,
        interview,
        care_options
    ]])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)   # multi-class probabilities

    # ✅ Pick class with highest probability
    pred_class = int(np.argmax(prediction[0]))

    # ✅ Map class to result
    if pred_class == 0:
        result = "No Treatment Needed"
    elif pred_class == 1:
        result = "Needs Treatment"
    else:
        result = "Further Evaluation Recommended"

    confidence = round(np.max(prediction[0]) * 100, 2)

    # 🎨 Confidence-based colors
    if confidence > 70:
        st.success(f"🩺 Prediction: {result} (Confidence: {confidence}%)")
    elif confidence > 40:
        st.warning(f"🩺 Prediction: {result} (Confidence: {confidence}%)")
    else:
        st.error(f"🩺 Prediction: {result} (Confidence: {confidence}%)")



import os
print(os.getcwd())