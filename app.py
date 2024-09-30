import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



df=pd.read_csv("diabetic_data.csv")
print(df.shape)



# Select rows to drop based on conditions
drop_indices = set(
    df[(df['diag_1'] == '?') & (df['diag_2'] == '?') & (df['diag_3'] == '?')].index
)
drop_indices = drop_indices.union(
    set(df['diag_1'][df['diag_1'] == '?'].index)
)
drop_indices = drop_indices.union(
    set(df['diag_2'][df['diag_2'] == '?'].index)
)
drop_indices = drop_indices.union(
    set(df['diag_3'][df['diag_3'] == '?'].index)
)
drop_indices = drop_indices.union(
    set(df['race'][df['race'] == '?'].index)
)
drop_indices = drop_indices.union(
    set(df['gender'][df['gender'] == 'Unknown/Invalid'].index)
)

# Drop the selected rows
df = df.drop(drop_indices)

print(len(df))




# Drop rows with NaN values from df
df.dropna(inplace = True)
# Print the total number of rows in df
print('Total data = ', len(df))
# Import numpy library for using its unique function
import numpy as np
# Print the total number of unique 'patient_nbr' in df
print('Unique entries = ', len(np.unique(df['patient_nbr'])))
# Remove duplicates based on 'patient_nbr' and keep the first occurrence
df.drop_duplicates(['patient_nbr'], keep = 'first', inplace = True)
# Print the total number of rows in df after removing duplicates
print('Length after removing Duplicates:', len(df))


import pandas as pd

# # 3. Filter by Length of Stay
df = df[(df['time_in_hospital'] >= 1) & (df['time_in_hospital'] <= 14)]

# # 4. Ensure Laboratory Tests were performed
df = df[df['num_lab_procedures'] > 0]

# # 5. Ensure Medications were administered
df = df[df['num_medications'] > 0]

# Check the number of remaining encounters
print(len(df))  # This should print 8,756 if the data and criteria match your description.


def safe_float_conversion(val):
    try:
        return float(val)
    except ValueError:  # if conversion fails
        return None  # or np.nan if you imported numpy

df['diag_1'] = df['diag_1'].map(safe_float_conversion)


df =df[df['diag_1'].between(250.00, 250.99)]
print(len(df))


# Drop columns
columns_to_drop = [ 'encounter_id', 'patient_nbr','payer_code',
    'chlorpropamide', 'acetohexamide', 'tolbutamide',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone',
    'diag_2', 'diag_3', 'weight', 'medical_specialty'
]

df.drop(columns=columns_to_drop, inplace=True)

df = df.loc[~df.discharge_disposition_id.isin([11,13,14,19,20,21])]



# Print unique values in 'age' column
print(np.unique(df['age']))

# Define a dictionary for age range replacements
replaceDict = {
    '[0-10)' : 5,
    '[10-20)' : 15,
    '[20-30)' : 25,
    '[30-40)' : 35,
    '[40-50)' : 45,
    '[50-60)' : 55,
    '[60-70)' : 65,
    '[70-80)' : 75,
    '[80-90)' : 85,
    '[90-100)' : 95
}

# Replace age ranges with the mean value of each range using the replaceDict
df['age'] = df['age'].apply(lambda x : replaceDict[x])

# Print the first 5 rows of the 'age' column after the transformation
print(df['age'].head())


# Reclassify 'discharge_disposition_id'
df['discharge_disposition_id'] = df['discharge_disposition_id'].apply(lambda x : 1 if int(x) in [6, 8, 9, 13]
                                                                           else ( 2 if int(x) in [3, 4, 5, 14, 22, 23, 24]
                                                                           else ( 10 if int(x) in [12, 15, 16, 17]
                                                                           else ( 11 if int(x) in [19, 20, 21]
                                                                           else ( 18 if int(x) in [25, 26]
                                                                           else int(x) )))))
# Reclassify 'admission_type_id'
df['admission_type_id'] = df['admission_type_id'].apply(lambda x : 1 if int(x) in [2, 7]
                                                        else ( 5 if int(x) in [6, 8]
                                                        else int(x) ))

# admission_source_id'
df['admission_source_id'] = df['admission_source_id'].apply(lambda x : 1 if int(x) in [2, 3]
                                                            else ( 4 if int(x) in [5, 6, 10, 22, 25]
                                                            else ( 9 if int(x) in [15, 17, 20, 21]
                                                            else ( 11 if int(x) in [13, 14]
                                                            else int(x) ))))


# Define the list of numerical columns
num_col = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
           'num_medications', 'number_outpatient', 'number_emergency',
           'number_inpatient', 'number_diagnoses']
# , 'patient_service', 'med_change', 'num_med'
# Initialize a new DataFrame to store statistics
statdataframe = pd.DataFrame()

# Add the column names to the DataFrame
statdataframe['numeric_column'] = num_col

# Initialize lists to store the statistics
skew_before = []
skew_after = []
kurt_before = []
kurt_after = []
standard_deviation_before = []
standard_deviation_after = []
log_transform_needed = []
log_type = []

# For each column in the list of numerical columns
for i in num_col:
    # Compute skewness before transformation
    skewval = df[i].skew()
    skew_before.append(skewval)

    # Compute kurtosis before transformation
    kurtval = df[i].kurtosis()
    kurt_before.append(kurtval)

    # Compute standard deviation before transformation
    sdval = df[i].std()
    standard_deviation_before.append(sdval)

    # If skewness and kurtosis are high, transformation is needed
    if (abs(skewval) >2) & (abs(kurtval) >2):
        log_transform_needed.append('Yes')

        # If the proportion of 0 values is less than 2%, apply log transformation
        if len(df[df[i] == 0])/len(df) <=0.02:
            log_type.append('log')
            skewvalnew = np.log(pd.DataFrame(df[df[i] > 0])[i]).skew()
            skew_after.append(skewvalnew)

            kurtvalnew = np.log(pd.DataFrame(df[df[i] > 0])[i]).kurtosis()
            kurt_after.append(kurtvalnew)

            sdvalnew = np.log(pd.DataFrame(df[df[i] > 0])[i]).std()
            standard_deviation_after.append(sdvalnew)

        # If the proportion of 0 values is more than 2%, apply log1p transformation
        else:
            log_type.append('log1p')
            skewvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).skew()
            skew_after.append(skewvalnew)

            kurtvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).kurtosis()
            kurt_after.append(kurtvalnew)

            sdvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).std()
            standard_deviation_after.append(sdvalnew)

    # If skewness and kurtosis are not high, no transformation is needed
    else:
        log_type.append('NA')
        log_transform_needed.append('No')

        skew_after.append(skewval)
        kurt_after.append(kurtval)
        standard_deviation_after.append(sdval)

# Add all the computed statistics to the DataFrame
statdataframe['skew_before'] = skew_before
statdataframe['kurtosis_before'] = kurt_before
statdataframe['standard_deviation_before'] = standard_deviation_before
statdataframe['log_transform_needed'] = log_transform_needed
statdataframe['log_type'] = log_type
statdataframe['skew_after'] = skew_after
statdataframe['kurtosis_after'] = kurt_after
statdataframe['standard_deviation_after'] = standard_deviation_after

# Print the DataFrame
# statdataframe


# If the log transform is needed according to our stats dataframe, apply the transformation
for i in range(len(statdataframe)):
    if statdataframe['log_transform_needed'][i] == 'Yes':
        # Get the column name
        colname = str(statdataframe['numeric_column'][i])

        # Apply the appropriate log transformation
        if statdataframe['log_type'][i] == 'log':
            df = df[df[colname] > 0]
            df[colname + "_log"] = np.log(df[colname])

        elif statdataframe['log_type'][i] == 'log1p':
            df = df[df[colname] >= 0]
            df[colname + "_log1p"] = np.log1p(df[colname])

# Drop some of the original columns that are not needed anymore
df = df.drop(['number_outpatient', 'number_inpatient', 'number_emergency'], axis = 1)

# print(df.shape)
# df.head()

# Import the necessary library
import scipy as sp

# Select the numeric columns
num_cols = ['age', 'time_in_hospital', 'num_lab_procedures',
       'num_procedures', 'num_medications', 'number_diagnoses']

# Keep only the rows in the dataframe that have a z-score less than 3 (i.e., remove outliers that are 3 standard deviations away from the mean)
df = df[(np.abs(sp.stats.zscore(df[num_cols])) < 3).all(axis=1)]

# Print the updated shape and head of the dataframe
# print(df.shape)
# df.head()


# Convert 'readmitted' to binary
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

df = df.drop(columns=['insulin'])


common_drugs = ['metformin', 'repaglinide', 'nateglinide', 'glimepiride', 'glipizide',
                'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'glyburide-metformin']


# Combine common and rare drugs lists
drugs = common_drugs

# Apply binary  for each drug
for drug in drugs:
    name = "take_" + drug
    df[name] = df[drug].isin(["Down", "Steady", "Up"]).astype(int)

# Remove the previous drug columns
df = df.drop(drugs, axis=1)

# Print the updated DataFrame
# print(df)

# Convert 'race' column into dummy/indicator variables
df = pd.get_dummies(df, columns = ["race"], prefix = "race", drop_first=True)
# Apply one-hot encoding to 'gender' column
df = pd.get_dummies(df, columns=['gender'], prefix = "gender", drop_first=True)



df = pd.get_dummies(df, columns=['A1Cresult'], drop_first=False)
# Drop 'A1Cresult' and 'A1C_None' columns from DataFrame 'df'
df = df.drop(["A1Cresult_None"], axis = 1)



df = pd.get_dummies(df, columns=['max_glu_serum'], drop_first=False)
# Drop 'A1Cresult' and 'A1C_None' columns from DataFrame 'df'
df = df.drop(["max_glu_serum_None"], axis = 1)


# Update the 'change' column to boolean values
df.loc[df.change == "Ch", "change"] = True
df.loc[df.change == "No", "change"] = False
df['change'] = df['change'].astype(int)  # Convert boolean values to integers (0 or 1)

# Update the 'diabetesMed' column to boolean values
df.loc[df.diabetesMed == "Yes", "diabetesMed"] = True
df.loc[df.diabetesMed == "No", "diabetesMed"] = False
df['diabetesMed'] = df['diabetesMed'].astype(int)  # Convert boolean values to integers (0 or 1)


target_counts = df['readmitted'].value_counts()
print(target_counts)



import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import pickle

# Assuming 'df' is your DataFrame and 'readmitted' is the target variable
# df = YOUR_DATAFRAME_HERE

# Use only the 10 specified features
selected_features = [
    'num_medications',
    'num_lab_procedures',
    'diag_1',
    'time_in_hospital',
    'number_inpatient_log1p',
    'number_diagnoses',
    'age',
    'num_procedures',
    'discharge_disposition_id',
    'admission_source_id'
]
X = df[selected_features]
y = df['readmitted']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Random Under-Sampling
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# Train Random Forest Classifier
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
clf.fit(X_resampled, y_resampled)

# Save the trained model
with open('RandomForest_Undersampling_model_10_features.pkl', 'wb') as file:
    pickle.dump(clf.best_estimator_, file)

                
        # Your performance evaluation code can go here.


# import streamlit as st
# import numpy as np
# import pickle

# # Load the trained model
# with open('RandomForest_Undersampling_model_10_features.pkl', 'rb') as file:
#     model = pickle.load(file)

# # Title
# st.title('🏥 Predicting Hospital Readmission')

# # Collect user input for top 10 features
# num_medications = st.number_input('Number of medications:', min_value=0)
# num_lab_procedures = st.number_input('Number of lab procedures:', min_value=0)
# diag_1 = st.selectbox('Primary diagnosis (diag_1):', ['Yes (250 to 250.99)', 'No'])
# time_in_hospital = st.selectbox('Time in hospital:', ['Yes (1 to 14)', 'No'])
# discharge_disposition_id = st.selectbox('Discharge disposition ID:', ['Home care', 'Transfer', 'Outpatients', 'Expired Home/Medical', 'Undefined'])
# admission_source_id = st.selectbox('Admission source ID:', ['Referral', 'Transfer', 'Undefined', 'Newborn'])
# number_inpatient_log1p = st.number_input('Number of inpatient events:', min_value=0)
# number_diagnoses = st.number_input('Number of diagnoses:', min_value=0)
# age = st.number_input('Age:', min_value=0)
# num_procedures = st.number_input('Number of procedures:', min_value=0)

# # Prepare data
# user_data = np.array([
#     num_medications,
#     num_lab_procedures,
#     1 if diag_1 == 'Yes (250 to 250.99)' else 0,
#     1 if time_in_hospital == 'Yes (1 to 14)' else 0,
#     number_inpatient_log1p,
#     number_diagnoses,
#     age,
#     num_procedures,
#     1 if discharge_disposition_id == 'Home care' else 2 if discharge_disposition_id == 'Transfer' else 10 if discharge_disposition_id == 'Outpatients' else 11 if discharge_disposition_id == 'Expired Home/Medical' else 18,
#     1 if admission_source_id == 'Referral' else 4 if admission_source_id == 'Transfer' else 9 if admission_source_id == 'Undefined' else 11
# ]).reshape(1, -1)

# # Add a button for predictions
# if st.button("Submit"):  # When this button is clicked, the following code will be executed.
#     # Make prediction
#     prediction_proba = model.predict_proba(user_data)
#     prob_of_readmission = prediction_proba[0][1]  # Probability of being readmitted
    
#     # Show prediction
#     st.markdown(f'<span style="color:red">📊 The model predicts a {prob_of_readmission * 100:.2f}% probability of being readmitted.</span>', unsafe_allow_html=True)




