import os
from pathlib import Path
import pandas as pd

def load_and_prepare_data():
    # Set path to folder with data
    folder_path = Path('data')

    # Set path to patient info CSV file
    patients_info_path = Path('data_patients/patients_info.csv')

    # Check folder existence
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory {folder_path} does not exist.")

    # Loading patient data
    patients_info = pd.read_csv(patients_info_path)
    print("Unique person_ids in patients_info:", patients_info['person_id'].nunique())

    # Load all time series CSV files
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    data_list = []
    for file in all_files:
        temp_data = pd.read_csv(file, delimiter=';')
        # Extract person_id from filename
        temp_data['person_id'] = os.path.splitext(os.path.basename(file))[0]
        data_list.append(temp_data)

    # Combine all data into one DataFrame
    data = pd.concat(data_list, ignore_index=True)

    # Remove duplicates
    data.drop_duplicates(inplace=True)

    # Time Conversion
    data['time'] = pd.to_datetime(data['time'])
    data['minute'] = data['time'].dt.minute
    data['hour_of_day'] = data['time'].dt.hour
    data['month'] = data['time'].dt.month
    data['day'] = data['time'].dt.day

    # Checking for anomalies (optional)
    data = data[(data['heart_rate'] > 40) & (data['heart_rate'] < 200)]

    # Filling in the blanks with the average value for selected columns
    for col in ['glucose', 'calories', 'heart_rate']:
        data[col] = data[col].fillna(data[col].mean())
    data = data.fillna(0)

    # Save combined data from 25 datasets
    data.to_csv('data_processed/combined_patientID_data.csv', index=False, sep=',')
    print('Combined data was saved as combined_data.csv')

    # Merge patient clinical data with time series data by person_id
    data = data.merge(patients_info, on='person_id', how='left')

    # Save processed data
    data.to_csv('data_processed/prepared_data_with_patient_info.csv', index=False, sep=',')
    print('Combined data was saved as prepared_data_with_patient_info.csv')

    # Check for empty values in the target variable
    if data['basal_rate'].isna().sum() > 0:
        raise ValueError("Target variable 'basal_rate' contains missing values!")

    # One-Hot Encoding for categorical variables (gender, treatment)
    data = pd.get_dummies(data, columns=['gender', 'treatment'], drop_first=False)

    # Checking for gaps after merging
    if data['HbA1c'].isna().sum() > 0:
        print("Warning: Missing values in clinical data after merge!")

    # Output of the number of unique person_id before One-Hot Encoding
    print(f'Quantity of datasets: {data["person_id"].nunique()}')

    # List of features
    features = [
        'glucose', 'calories', 'heart_rate', 'steps',
        'bolus_volume_delivered', 'carb_input', 'minute', 'hour_of_day', 'month', 'day',
        'HbA1c', 'age', 'dx_time', 'weight', 'height'
    ]

    ## Add One-Hot Encoding columns for gender and treatment
    features += [col for col in data.columns if col.startswith('gender_') or col.startswith('treatment_')]

    X = data[features]
    y = data['basal_rate']
    return X, y
