import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

def calculate_correction_insulin(glucose, target_glucose=120, correction_factor=50):
    """Розрахунок інсуліну на корекцію рівня глюкози."""
    return max(0, (glucose - target_glucose) / correction_factor)

data['insulin_dose'] = data['glucose'].apply(calculate_correction_insulin)

X = data[['glucose', 'calories', 'heart_rate', 'steps', 'basal_rate', 'bolus_volume_delivered']]
y = data['insulin_dose']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

data['insulin_dose_rf'] = rf_model.predict(X_scaled)

data_filtered = data[data['insulin_dose_rf'] > 0]

average_doses = data_filtered.groupby('patient_id')['insulin_dose_rf'].mean().reset_index()

average_doses.to_csv("data/average_insulin_doses.csv", index=False)

print("\nСередні дози інсуліну для 10 пацієнтів:")
print(average_doses.head(10))

new_patient_data = pd.DataFrame({
    'glucose': [150],
    'calories': [2000],
    'heart_rate': [80],
    'steps': [50],
    'basal_rate': [0.05],
    'bolus_volume_delivered': [0]
})

new_patient_data_scaled = scaler.transform(new_patient_data)

new_patient_dose = rf_model.predict(new_patient_data_scaled)

print("\nПрогнозована доза інсуліну для нового пацієнта:", new_patient_dose[0])
