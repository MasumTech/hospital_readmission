# Hospital Readmission Prediction

A machine-learning web application that estimates the probability of hospital readmission for patients with diabetes. The project combines a trained Random Forest classifier with an interactive Streamlit interface.

> **Portfolio project:** This application is for educational and demonstration purposes only. It is not a clinical decision-support tool or a substitute for professional medical advice.

## What the application does

The Streamlit interface collects ten patient and encounter features and returns the model's estimated probability of readmission:

- number of medications
- number of laboratory procedures
- primary diagnosis indicator
- time in hospital
- number of previous inpatient encounters
- number of diagnoses
- age
- number of procedures
- discharge disposition
- admission source

## Technical approach

The training workflow in `app.py` includes:

1. data-quality checks and removal of incomplete or invalid records
2. patient-level duplicate removal
3. feature engineering and categorical encoding
4. outlier handling and transformations for skewed numeric variables
5. class balancing with random under-sampling
6. model selection using `GridSearchCV`
7. training a Random Forest classifier on ten selected features
8. serialising the trained estimator for use by the Streamlit application

## Tech stack

- Python
- pandas and NumPy
- scikit-learn
- imbalanced-learn
- SciPy
- Streamlit
- Matplotlib and Seaborn

## Repository structure

```text
.
├── Hospital_Readmitted.py
├── RandomForest_Undersampling_model_10_features.pkl
├── app.py
├── demo.py
└── requirements.txt
```

- `Hospital_Readmitted.py` — interactive Streamlit prediction interface
- `app.py` — preprocessing, feature engineering, training, and model export
- `demo.py` — additional application/demo implementation
- `RandomForest_Undersampling_model_10_features.pkl` — trained model used by the interface
- `requirements.txt` — Python dependencies

## Run the prediction app locally

### 1. Clone the repository

```bash
git clone https://github.com/MasumTech/hospital_readmission.git
cd hospital_readmission
```

### 2. Create and activate a virtual environment

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start Streamlit

```bash
streamlit run Hospital_Readmitted.py
```

The application will open in your browser on Streamlit's local development server.

## Retraining notes

The prediction interface can run with the model artifact already included in this repository. Retraining with `app.py` additionally requires the source dataset as `diabetic_data.csv` in the project root.

## Current limitations

- the repository does not include the source training dataset
- model performance metrics and experiment tracking are not yet published here
- the serialised model depends on compatible Python and scikit-learn versions
- the interface is a portfolio demonstration and has not been validated for clinical use

## Planned improvements

- add automated tests for preprocessing and inference
- move preprocessing and model training into reusable modules
- publish evaluation metrics and a confusion matrix
- add Docker support and continuous integration
- deploy a live demonstration

## Author

**Md. Masum Reza**  
Python / Django Backend Developer · MSc Data Science (Merit)

- GitHub: [MasumTech](https://github.com/MasumTech)

