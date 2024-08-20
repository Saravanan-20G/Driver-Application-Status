import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model
model = joblib.load('rf_model.pkl')

# Load training data to get feature names
df_train = pd.read_csv(r'C:\Users\Saravanan\OneDrive\Desktop\drive\TLC_New_Driver_Application_Status.csv')
expected_columns = df_train.columns.tolist()

# Define a function to predict the status
def predict_status(input_data):
    # Convert input data to DataFrame and ensure it has the correct columns
    input_df = pd.DataFrame([input_data], columns=expected_columns)
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit UI
st.title("TLC New Driver Application Status Prediction")
st.write("Predict the application status based on the inputs")

# Creating input fields for each feature
app_year = st.number_input('Application Year', min_value=2000, max_value=2025, value=2023)
app_month = st.number_input('Application Month', min_value=1, max_value=12, value=1)
app_day = st.number_input('Application Day', min_value=1, max_value=31, value=1)
processing_time = st.number_input('Processing Time (days)', min_value=0, max_value=365, value=30)

# Add all necessary input fields
drug_test = st.selectbox('Drug Test Completed?', ['Yes', 'No'])
wav_course = st.selectbox('WAV Course Completed?', ['Yes', 'No'])
defensive_driving = st.selectbox('Defensive Driving Completed?', ['Yes', 'No'])
driver_exam = st.selectbox('Driver Exam Completed?', ['Yes', 'No'])
medical_clearance_form = st.selectbox('Medical Clearance Form Submitted?', ['Yes', 'No'])

# Add any other fields that are part of the model training
app_no = st.text_input('Application Number', '')
fru_interview_scheduled = st.selectbox('FRU Interview Scheduled?', ['Yes', 'No'])
last_updated_year = st.number_input('Last Updated Year', min_value=2000, max_value=2025, value=2023)

# Prepare the input data for prediction
input_data = {
    'App Year': app_year,
    'App Month': app_month,
    'App Day': app_day,
    'Processing Time': processing_time,
    'Drug Test': 1 if drug_test == 'Yes' else 0,
    'WAV Course': 1 if wav_course == 'Yes' else 0,
    'Defensive Driving': 1 if defensive_driving == 'Yes' else 0,
    'Driver Exam': 1 if driver_exam == 'Yes' else 0,
    'Medical Clearance Form': 1 if medical_clearance_form == 'Yes' else 0,
    'App No': app_no,
    'FRU Interview Scheduled': 1 if fru_interview_scheduled == 'Yes' else 0,
    'Last Updated Year': last_updated_year,
    # Include any other features that were part of the training dataset
}

# When the user clicks 'Predict'
if st.button('Predict'):
    prediction = predict_status(input_data)
    if prediction == 1:  # Assuming 1 = Approved, 0 = Rejected
        st.success('The application is likely to be Approved.')
    else:
        st.error('The application is likely to be Rejected.')

# Optional: Display the input data for verification
st.write("Input Data:", input_data)
