import streamlit as st
import numpy as np
import joblib

# Load pre-trained models
diabetes_model = joblib.load("C:\\Users\\Pavani\\Desktop\\Disease_Prediction\\diabetes_model .pkl")
heart_model = joblib.load("C:\\Users\\Pavani\\Desktop\\Disease_Prediction\\heartdisease_model .pkl")
parkinsons_model = joblib.load("C:\\Users\\Pavani\\Desktop\\Disease_Prediction\\parkinsonsdisease_model .pkl")

# Title
st.title("Disease Prediction Web App")

# Sidebar for navigation
st.sidebar.title("Multiple Disease Prediction")
disease_option = st.sidebar.radio("Choose one disease to predict", ["Diabetes", "Heart Disease", "Parkinson's"])

# Diabetes Prediction
if disease_option == "Diabetes":
    st.header("Diabetes Prediction")
    glucose = st.number_input("Glucose Level", 0, 200, 120)
    bp = st.number_input("Blood Pressure", 0, 150, 70)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 0, 120, 30)
    
    if st.button("Predict Diabetes"):
        data = np.array([[glucose, bp, bmi, dpf, age]])
        result = diabetes_model.predict(data)
        st.success("Diabetes Prediction: " + ("Positive" if result[0] == 1 else "Negative"))

# Heart Disease Prediction
elif disease_option == "Heart Disease":
    st.header("Heart Disease Prediction")
    age = st.number_input("Age", 0, 120, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 1)
    thalach = st.number_input("Max Heart Rate", 0, 220, 150)
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
    ca = st.number_input("Major Vessels Colored (0-4)", 0, 4, 0)
    
    if st.button("Predict Heart Disease"):
        sex = 1 if sex == "Male" else 0
        data = np.array([[age, sex, cp, thalach, oldpeak, ca]])
        result = heart_model.predict(data)
        st.success("Heart Disease Prediction: " + ("Positive" if result[0] == 1 else "Negative"))

# Parkinson's Disease Prediction
elif disease_option == "Parkinson's":
    st.header("Parkinson's Disease Prediction")
    fo = st.number_input("MDVP:Fo(Hz)", 50.0, 300.0, 150.0)
    shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0, 0.01)
    hnr = st.number_input("HNR", 0.0, 50.0, 20.0)
    rpde = st.number_input("RPDE", 0.0, 1.0, 0.5)
    spread1 = st.number_input("Spread1", -8.0, 0.0, -4.0)
    ppe = st.number_input("PPE", 0.0, 1.0, 0.1)
    
    if st.button("Predict Parkinson's"):
        data = np.array([[fo, shimmer, hnr, rpde, spread1, ppe]])
        result = parkinsons_model.predict(data)
        st.success("Parkinson's Prediction: " + ("Positive" if result[0] == 1 else "Negative"))

