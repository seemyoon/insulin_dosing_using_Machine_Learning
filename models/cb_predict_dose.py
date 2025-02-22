from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_preparation.data_preparation import load_and_prepare_data
from pathlib import Path

# Set path to folder with data
folder_path = Path('data')

# Prepare data. From data_preparation.py
X_scaled, y = load_and_prepare_data(folder_path)

# Split into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# CatBoost model creation and training
model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, random_state=42, verbose=200)
model.fit(X_train, y_train)

# Prediction and model evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print(f'CatBoost Regression. Mean Squared Error: {mse}')
print(f'CatBoost Regression. R²: {r2}')