# import streamlit as st
# import numpy as np
# import pickle

# # Initialize session state for widgets
# if 'init' not in st.session_state:
#     st.session_state.init = True
#     st.session_state.num_medications = 1
#     st.session_state.num_lab_procedures = 1
#     st.session_state.diag_1 = 'Yes (250 to 250.99)'
#     st.session_state.time_in_hospital = 'Yes (1 to 14)'
#     st.session_state.number_inpatient_log1p = 0
#     st.session_state.number_diagnoses = 1
#     st.session_state.age = 15
#     st.session_state.num_procedures = 0
#     st.session_state.discharge_disposition_id = 'Home care'
#     st.session_state.admission_source_id = 'Referral'

# # Load the trained model
# try:
#     with open('RandomForest_Undersampling_model_10_features.pkl', 'rb') as file:
#         model = pickle.load(file)
# except FileNotFoundError:
#     st.error("Model file not found. Make sure the RandomForest_Undersampling_model_10_features.pkl file is in the same directory as this script.")

# # Custom styles
# st.markdown("""
#     <style>
#         .title-box {
#             background-color: #900C3F;
#             color: white;
#             padding: 10px;
#             border-radius: 5px;
#         }
#     </style>
#     """, unsafe_allow_html=True)

# # Title
# st.markdown("""
#     <div class="title-box">
#         <h1>🏥 Predicting Hospital Readmission For Diabetic Patients</h1>
#     </div>
#     """, unsafe_allow_html=True)

# # Create widgets using session_state variables
# st.session_state.num_medications = st.number_input('Number of medications:', value=st.session_state.num_medications, help="Range: 1-81")
# st.session_state.num_lab_procedures = st.number_input('Number of lab procedures:', value=st.session_state.num_lab_procedures, help="Range: 1-132")
# st.session_state.diag_1 = st.selectbox('Primary diagnosis (diag_1):', ['Yes (250 to 250.99)', 'No'], index=['Yes (250 to 250.99)', 'No'].index(st.session_state.diag_1))
# st.session_state.time_in_hospital = st.selectbox('Time in hospital:', ['Yes (1 to 14)', 'No'], index=['Yes (1 to 14)', 'No'].index(st.session_state.time_in_hospital))
# st.session_state.number_inpatient_log1p = st.number_input('Number of inpatient events:', value=st.session_state.number_inpatient_log1p, help="Range: 0-21")
# st.session_state.number_diagnoses = st.number_input('Number of diagnoses:', value=st.session_state.number_diagnoses, help="Range: 1-16")
# st.session_state.age = st.number_input('Age:', value=st.session_state.age, help="Range: 15-95")
# st.session_state.num_procedures = st.number_input('Number of procedures:', value=st.session_state.num_procedures, help="Range: 0-6")
# st.session_state.discharge_disposition_id = st.selectbox('Discharge disposition ID:', ['Home care', 'Transfer', 'Outpatients', 'Expired Home/Medical', 'Undefined'], index=['Home care', 'Transfer', 'Outpatients', 'Expired Home/Medical', 'Undefined'].index(st.session_state.discharge_disposition_id))
# st.session_state.admission_source_id = st.selectbox('Admission source ID:', ['Referral', 'Transfer', 'Undefined', 'Newborn'], index=['Referral', 'Transfer', 'Undefined', 'Newborn'].index(st.session_state.admission_source_id))

# # Submit button
# if st.button("Submit"):
#     # Prepare data and do your predictions here
#     user_data = np.array([
#         st.session_state.num_medications,
#         st.session_state.num_lab_procedures,
#         1 if st.session_state.diag_1 == 'Yes (250 to 250.99)' else 0,
#         1 if st.session_state.time_in_hospital == 'Yes (1 to 14)' else 0,
#         st.session_state.number_inpatient_log1p,
#         st.session_state.number_diagnoses,
#         st.session_state.age,
#         st.session_state.num_procedures,
#         # Add logic for discharge_disposition_id and admission_source_id
#     ]).reshape(1, -1)

#     prediction_proba = model.predict_proba(user_data)
#     prob_of_readmission = prediction_proba[0][1]
#     st.write(f"The model predicts a {prob_of_readmission * 100:.2f}% probability of being readmitted.")

# # Reset button logic
# if st.button('Reset'):
#     st.session_state.init = True
#     st.session_state.num_medications = 1
#     st.session_state.num_lab_procedures = 1
#     st.session_state.diag_1 = 'Yes (250 to 250.99)'
#     st.session_state.time_in_hospital = 'Yes (1 to 14)'
#     st.session_state.number_inpatient_log1p = 0
#     st.session_state.number_diagnoses = 1
#     st.session_state.age = 15



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
    st.error("Model file not found. Make sure you place the model file in the same directory as this script.")

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
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("""
    <div class="title-box">
        <h1>🏥 Predicting Hospital Readmission For Diabetic Patients</h1>
    </div>
    """, unsafe_allow_html=True)

# Initialize a flag for valid input
valid_input = True

# Collect user input with manual validation
num_medications = st.number_input('Number of medications:', value=1, help="Range: 1-81")
num_lab_procedures = st.number_input('Number of lab procedures:', value=1, help="Range: 1-132")
diag_1 = st.selectbox('Primary diagnosis (diag_1):', ['Yes (250 to 250.99)', 'No'])
time_in_hospital = st.selectbox('Time in hospital:', ['Yes (1 to 14)', 'No'])
discharge_disposition_id = st.selectbox('Discharge disposition ID:', ['Home care', 'Transfer', 'Outpatients', 'Expired Home/Medical', 'Undefined'])
admission_source_id = st.selectbox('Admission source ID:', ['Referral', 'Transfer', 'Undefined', 'Newborn'])
number_inpatient_log1p = st.number_input('Number of inpatient events:', value=0, help="Range: 0-21")
number_diagnoses = st.number_input('Number of diagnoses:', value=1, help="Range: 1-16")
age = st.number_input('Age:', value=15, help="Range: 15-95")
num_procedures = st.number_input('Number of procedures:', value=0, help="Range: 0-6")

# Check if inputs are valid
if st.button("Submit"):
    st.experimental_rerun()
    if valid_input:
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
            1 if discharge_disposition_id == 'Home care' else 2 if discharge_disposition_id == 'Transfer' else 3,
            1 if admission_source_id == 'Referral' else 2 if admission_source_id == 'Transfer' else 3
        ]).reshape(1, -1)

        # Make prediction
        prediction_proba = model.predict_proba(user_data)
        prob_of_readmission = prediction_proba[0][1]  # Probability of being readmitted

        # Show prediction
        st.markdown(f"""
            <div class="result-box">
                <span>📊 The model predicts a {prob_of_readmission * 100:.2f}% probability of being readmitted.</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.session_state.reset = True

# Show the Reset button and functionality to reset the page
if st.button("Reset"):
    print("Reset button clicked")  # This should appear in your terminal
    st.experimental_rerun()

st.markdown("""
    <button id="btn">Click Me</button>
    <p id="output"></p>
    
    <script>
        document.getElementById("btn").addEventListener("click", function() {
            document.getElementById("output").innerHTML = "Button clicked!";
        });
    </script>
""", unsafe_allow_html=True)