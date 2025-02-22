from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_preparation.data_preparation import load_and_prepare_data
from pathlib import Path

# Set path to folder with data
folder_path = Path('data')

# Prepare data. From data_preparation.py
X_scaled, y = load_and_prepare_data(folder_path)

# Split into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# DT model creation and training
model = DecisionTreeRegressor(max_depth=12, min_samples_leaf=2, min_samples_split=2, random_state=42)
model.fit(X_train, y_train)

# Prediction and model evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print(f'Decision tree. Mean Squared Error: {mse}')
print(f'Decision tree. R²: {r2}')
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_preparation.data_preparation import load_and_prepare_data
from pathlib import Path

# Set path to folder with data
folder_path = Path('data')

# Prepare data. From data_preparation.py
X_scaled, y = load_and_prepare_data(folder_path)

# Split into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the model
model = DecisionTreeRegressor(random_state=42)

# Define the parameter grid
param_grid = {
    'max_depth': [5, 8, 10, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=8, scoring='neg_mean_squared_error')

# Fit the grid search
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
best_params = grid_search.best_params_
print("Best hyperparameters found by GridSearchCV:", best_params)

# Evaluate the model with the best parameters
best_model = grid_search.best_estimator_

# Prediction and model evaluation
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print(f'Decision Tree (with best parameters). Mean Squared Error: {mse}')
print(f'Decision Tree (with best parameters). R²: {r2}')
