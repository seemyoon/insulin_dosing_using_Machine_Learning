import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

folder_path = "data/"
all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

data_list = []
for file in all_files:
    temp_data = pd.read_csv(file, delimiter=';')
    temp_data['patient_id'] = os.path.basename(file).split('.')[0]
    data_list.append(temp_data)

data = pd.concat(data_list, ignore_index=True)

data['time'] = pd.to_datetime(data['time'])

data['time_of_day'] = data['time'].dt.hour
data['previous_glucose'] = data.groupby('patient_id')['glucose'].shift(1)
data['change_in_glucose'] = data['glucose'] - data['previous_glucose']

data.dropna(subset=['previous_glucose'], inplace=True)

def calculate_correction_insulin(glucose, target_glucose=120, correction_factor=50):
    return max(0, (glucose - target_glucose) / correction_factor)

data['insulin_dose'] = data['glucose'].apply(calculate_correction_insulin)

X = data[['glucose', 'calories', 'heart_rate', 'steps', 'basal_rate', 'bolus_volume_delivered', 'time_of_day', 'change_in_glucose']]
y_dose = data['insulin_dose']

X = pd.get_dummies(X, columns=[], drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_dose_train, y_dose_test = train_test_split(X_scaled, y_dose, test_size=0.2, random_state=42)

rf_dose_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_dose_model.fit(X_train, y_dose_train)

data['insulin_dose_rf'] = rf_dose_model.predict(X_scaled)

y_pred = rf_dose_model.predict(X_scaled)
mse = mean_squared_error(y_dose, y_pred)
rmse = mse ** 0.5
print(f'RMSE: {rmse}')

average_doses = data.groupby('patient_id')[['insulin_dose', 'insulin_dose_rf']].mean().reset_index()

result = average_doses

result.to_csv("result/insulin_doses.csv", index=False)

print(result.head(10))
