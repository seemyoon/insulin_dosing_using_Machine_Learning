import os

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory {folder_path} does not exist.")

    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

    data_list = []

    for file in all_files:
        temp_data = pd.read_csv(file, delimiter=';')
        data_list.append(temp_data)

    # Combine all data into one DataFrame
    data = pd.concat(data_list, ignore_index=True)

    # Time Conversion
    data['time'] = pd.to_datetime(data['time'])
    data['minute'] = data['time'].dt.minute
    data['hour_of_day'] = data['time'].dt.hour
    data['month'] = data['time'].dt.month
    data['day'] = data['time'].dt.day

    # Checking for anomalies (optional)
    data = data[(data['heart_rate'] > 40) & (data['heart_rate'] < 200)]

    # Filling in the blanks with the average value
    data['glucose'] = data['glucose'].fillna(data['glucose'].mean())
    data['calories'] = data['calories'].fillna(data['calories'].mean())
    data['heart_rate'] = data['heart_rate'].fillna(data['heart_rate'].mean())

    # Filling missing values with zeros (NaN)
    data = data.fillna(0)

    # Extraction of features (X) and target variable (y)
    x = data[['glucose',
              'calories',
              'heart_rate',
              'steps',
              'bolus_volume_delivered',
              'carb_input',
              'minute',
              'hour_of_day',
              'month',
              'day',
              ]]

    y = data['basal_rate']

    # Data scaling
    x_scaled = StandardScaler().fit_transform(x)

    return x_scaled, y
