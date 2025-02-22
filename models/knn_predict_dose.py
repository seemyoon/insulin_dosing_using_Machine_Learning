from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_preparation.data_preparation import load_and_prepare_data
from pathlib import Path

# Set path to folder with data
folder_path = Path('data')

# Prepare data. From data_preparation.py
X_scaled, y = load_and_prepare_data(folder_path)

# Split into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# KNN model creation
knn = KNeighborsRegressor()

# Define the parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  # Range of values for n_neighbors
    'weights': ['uniform', 'distance'],  # Different weight strategies
    'metric': ['euclidean', 'manhattan'],  # Different distance metrics
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# Fit GridSearchCV to find the best parameters
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Print best parameters
print("Best Parameters for KNN:", best_params)

# Prediction and model evaluation with best model
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print(f'K-nearest neighbors with optimized parameters. Mean Squared Error: {mse}')
print(f'K-nearest neighbors with optimized parameters. R²: {r2}')
