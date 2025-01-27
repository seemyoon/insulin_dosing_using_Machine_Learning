import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
import matplotlib.pyplot as plt

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

feature_importances = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=["importance"]).sort_values("importance", ascending=False)
print("\nВажливість ознак (Feature Importances):")
print(feature_importances)

correlation_matrix = data[['glucose', 'calories', 'heart_rate', 'steps', 'basal_rate', 'bolus_volume_delivered', 'insulin_dose']].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Кореляційна матриця")
plt.show()

y_pred = rf_model.predict(X_test)

average_doses = data.groupby('patient_id')[['insulin_dose', 'insulin_dose_rf']].mean().reset_index()
average_doses.to_csv("result/average_insulin_doses_by_patient.csv", index=False)

print("\nСередні дози інсуліну пацієнтам:")
print(average_doses.head())
