import os
import pandas as pd
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

if data.isnull().sum().any():
    print("Наявність пропусків у даних. Заповнення пропусків...")
    data = data.fillna(0)

data['glucose_lag_1'] = data.groupby('patient_id')['glucose'].shift(1)
data['glucose_lag_2'] = data.groupby('patient_id')['glucose'].shift(2)
data['glucose_lag_3'] = data.groupby('patient_id')['glucose'].shift(3)

data['change_in_glucose'] = data['glucose'] - data['glucose_lag_1']
data['time_of_day'] = data['time'].dt.hour
data.dropna(inplace=True)

def calculate_correction_insulin(glucose, carb_input, basal_rate, target_glucose=120, correction_factor=45):
    correction = max(0, (glucose - target_glucose) / correction_factor)
    carb_effect = carb_input / 10
    basal_effect = basal_rate / 2
    return correction + carb_effect - basal_effect

data['insulin_dose'] = data.apply(lambda row: calculate_correction_insulin(
    row['glucose'], row['carb_input'], row['basal_rate']), axis=1)

X = data[['glucose_lag_1', 'glucose_lag_2', 'glucose_lag_3', 'calories', 'heart_rate', 'steps', 'basal_rate', 'bolus_volume_delivered', 'carb_input', 'change_in_glucose', 'time_of_day']]
y = data['insulin_dose']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "SVM": SVR(kernel='rbf', C=10, gamma='scale'),
    "KNN": KNeighborsRegressor(n_neighbors=3, weights='distance'),
    "Random Forest": RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R²": r2}

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

plt.savefig("result_algorithms.png")

plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index, y=results_df["MSE"])
plt.title("Порівняння моделей за MSE", fontsize=16)
plt.ylabel("Mean Squared Error", fontsize=14)
plt.xlabel("Algorithms", fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)

for index, value in enumerate(results_df["MSE"]):
    plt.text(index, value + 0.5, f'{value:.2f}', ha='center', fontsize=12, color='black')

plt.savefig("result_algorithms_mse.png")
