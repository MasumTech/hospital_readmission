import streamlit as st
import numpy as np
import pickle

# Initialize session state for reset functionality
if 'reset' not in st.session_state:
    st.session_state.reset = False

# Load the trained model
try:
    with open('RandomForest_Undersampling_model_10_features.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Make sure the RandomForest_Undersampling_model_10_features.pkl file is in the same directory as this script.")

# Custom styles
st.markdown("""
    <style>
        .title-box {
            background-color: #900C3F;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        .result-box {
            background-color: #C70039;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        .input-box {
            margin-bottom: 15px;
            border: 1px solid #900C3F;
            border-radius: 5px;
            padding: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("""
    <div class="title-box" style="display: inline-block;">
        <h1 style="color: white; display: inline;"> Predicting Hospital Readmission For Diabetic Patients</h1>
    </div>
    """, unsafe_allow_html=True)
st.write("")  # This adds a line of space

# Initialize a flag for valid input
valid_input = True

# Collect user input with manual validation
num_medications = st.number_input('Number of medications:', value=1, help="Range: 1-81", min_value=1, max_value=81)
if not(1 <= num_medications <= 81):
    st.warning('Number of medications should be between 1 and 81.')
    valid_input = False

num_lab_procedures = st.number_input('Number of lab procedures:', value=1, help="Range: 1-132", min_value=1, max_value=132)
if not(1 <= num_lab_procedures <= 132):
    st.warning('Number of lab procedures should be between 1 and 132.')
    valid_input = False

diag_1 = st.selectbox('Primary diagnosis (diag_1):', ['Yes (250 to 250.99)', 'No'])
time_in_hospital = st.selectbox('Time in hospital:', ['Yes (1 to 14)', 'No'])
discharge_disposition_id = st.selectbox('Discharge disposition ID:', ['Home care', 'Transfer', 'Outpatients', 'Expired Home/Medical', 'Undefined'])
admission_source_id = st.selectbox('Admission source ID:', ['Referral', 'Transfer', 'Undefined', 'Newborn'])

number_inpatient_log1p = st.number_input('Number of inpatient events:', value=0, help="Range: 0-21", min_value=0, max_value=21)
if not(0 <= number_inpatient_log1p <= 21):
    st.warning('Number of inpatient events should be between 0 and 21.')
    valid_input = False

number_diagnoses = st.number_input('Number of diagnoses:', value=1, help="Range: 1-16", min_value=1, max_value=16)
if not(1 <= number_diagnoses <= 16):
    st.warning('Number of diagnoses should be between 1 and 16.')
    valid_input = False

age = st.number_input('Age:', value=15, help="Range: 15-95", min_value=15, max_value=95)
if not(15 <= age <= 95):
    st.warning('Age should be between 15 and 95.')
    valid_input = False

num_procedures = st.number_input('Number of procedures:', value=0, help="Range: 0-6", min_value=0, max_value=6)
if not(0 <= num_procedures <= 6):
    st.warning('Number of procedures should be between 0 and 6.')
    valid_input = False

# Check if inputs are valid
if valid_input:
    if st.button("Submit", key='submit'):
        # Prepare data
        user_data = np.array([
            num_medications,
            num_lab_procedures,
            1 if diag_1 == 'Yes (250 to 250.99)' else 0,
            1 if time_in_hospital == 'Yes (1 to 14)' else 0,
            number_inpatient_log1p,
            number_diagnoses,
            age,
            num_procedures,
            1 if discharge_disposition_id == 'Home care' else 2 if discharge_disposition_id == 'Transfer' else 10 if discharge_disposition_id == 'Outpatients' else 11 if discharge_disposition_id == 'Expired Home/Medical' else 18,
            1 if admission_source_id == 'Referral' else 4 if admission_source_id == 'Transfer' else 9 if admission_source_id == 'Undefined' else 11
        ]).reshape(1, -1)

        # Make prediction
        prediction_proba = model.predict_proba(user_data)
        prob_of_readmission = prediction_proba[0][1]  # Probability of being readmitted

        # Show prediction in colored box
        st.markdown(f"""
            <div class="result-box">
                <span> The model predicts a {prob_of_readmission * 100:.2f}% probability of being readmitted.</span>
            </div>
            """, unsafe_allow_html=True)

        # Show reset button
        st.session_state.reset = True

# Insert space before the Reset button
if st.session_state.reset:
    st.write("")  # This adds a line of space

    if st.button("Reset"):
        # Clear the inputs and any outputs
        for key in list(st.session_state.keys()):
            del st.session_state[key]  # Clear all keys in session state
        st.session_state.reset = False  # Reset the reset flag
