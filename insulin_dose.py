import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

data['normalized_heart_rate'] = (data['heart_rate'] - 60) / (180 - 60)

def calculate_correction_insulin(row, target_glucose=120, correction_factor=50):
    correction = max(0, (row['glucose'] - target_glucose) / correction_factor)
    carb_effect = row.get('carb_input', 0) / 10
    basal_effect = row.get('basal_rate', 0)
    bolus_effect = row.get('bolus_volume_delivered', 0)

    heart_rate_effect = row['normalized_heart_rate']
    calorie_effect = row['calories']

    steps_effect = (row['steps'] / 10000) * 0.18

    return max(0,
               correction + carb_effect - basal_effect - bolus_effect - heart_rate_effect - steps_effect - calorie_effect)


data['insulin_dose'] = data.apply(calculate_correction_insulin, axis=1)

X = data[['glucose', 'calories', 'basal_rate', 'bolus_volume_delivered', 'carb_input', 'time_of_day', 'heart_rate',
          'steps']].fillna(0)
y = data['insulin_dose']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "SVM": SVR(kernel='rbf', C=1, gamma='scale'),
    "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance'),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42)
}

trained_models = {name: model.fit(X_train, y_train) for name, model in models.items()}

results = {
    name: {"MSE": mean_squared_error(y_test, model.predict(X_test)), "R²": r2_score(y_test, model.predict(X_test))} for
    name, model in models.items()}
results_df = pd.DataFrame(results).T
print("Порівняння моделей за точністю:")
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index, y=results_df["R²"])
plt.title("Порівняння моделей за R²", fontsize=16)
plt.ylabel("R² Score", fontsize=14)
plt.xlabel("Algorithms", fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)

for index, value in enumerate(results_df["R²"]):
    plt.text(index, value + 0.005, f'{value:.4f}', ha='center', fontsize=12, color='black')

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/result_algorithms.png")

new_data = data.copy()
X_new = new_data[
    ['glucose', 'calories', 'basal_rate', 'bolus_volume_delivered', 'carb_input', 'time_of_day', 'heart_rate',
     'steps']].fillna(0)
X_new_scaled = scaler.transform(X_new)

for name, model in models.items():
    new_data[f'insulin_dose_{name}'] = np.maximum(0, model.predict(X_new_scaled))

patient_dose = new_data.groupby('patient_id')[[f'insulin_dose_{name}' for name in models.keys()]].mean()
output_path = "results/new_insulin_doses_all_models.csv"
patient_dose.to_csv(output_path, index=True)
print(patient_dose)