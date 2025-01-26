import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
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
X = data[['calories', 'heart_rate', 'steps', 'basal_rate', 'bolus_volume_delivered', 'carb_input']]
y = data['glucose']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "SVM": SVR(),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R²": r2}

results_df = pd.DataFrame(results).T
print("\nПорівняння моделей за точністю:")
print(results_df)

best_model_name = results_df['MSE'].idxmin()
print(f"\nНайточніша модель: {best_model_name}")
