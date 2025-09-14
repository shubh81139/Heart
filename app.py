import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler    



model = joblib.load("heart (1).pkl")
scaler = joblib.load("scaler.pkl")
expected_column = joblib.load("columns.pkl")


st.title("Heart Disease Prediction❤️")
st.markdown("Please provide the following information.")


age = st.slider('Age',18,100,40)
sex = st.selectbox('Sex', ['M', 'F'])
chest_pain = st.selectbox('Chest Pain Type', ['TA', 'ATA', 'NAP', 'ASY'])
resting_bp = st.number_input('Resting Blood Pressure (in mm Hg)', 80, 200, 120)   
cholesterol = st.number_input('Cholesterol (in mg/dl)', 100, 600, 200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No']) 
resting_ecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST', 'LVH']) 
max_hr = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
exercise_angina = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
oldpeak = st.slider('Oldpeak (ST depression induced by exercise relative to rest)', 0.0, 6.0, 1.0)
st_slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Up', 'Flat', 'Down'])


if st.button('Predict'):
    features = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == 'M' else 0],
        'chest_pain': [chest_pain],
        'resting_bp': [resting_bp],
        'cholesterol': [cholesterol],
        'fasting_bs': [1 if fasting_bs == 'Yes' else 0],
        'resting_ecg': [resting_ecg],
        'max_hr': [max_hr],
        'exercise_angina': [1 if exercise_angina == 'Yes' else 0],
        'oldpeak': [oldpeak],
        'slope': [st_slope]
    })

    for column in expected_column:
      if column not in features.columns:
        features[column] = 0
        
    features = features[expected_column]
        
    # Scale the features
    # Scaled_feature = scaler.transform(features)


    # Make prediction
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("You are at risk of heart disease.💔")
    else:
        st.success("You are not at risk of heart disease.😊")

st.subheader("Prediction Results")