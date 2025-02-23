from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from data_preparation.data_preparation import load_and_prepare_data

def predict_dose():
    # Set path to folder with data
    folder_path = Path('data')

    # Prepare data. From data_preparation.py
    X_scaled, y = load_and_prepare_data(folder_path)

    # Split into training and test samples
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # RF model creation and training with feature selection (max_features)
    model = KNeighborsRegressor(weights='distance', n_neighbors=7, metric='manhattan')
    model.fit(X_train, y_train)

    # Prediction and model evaluation
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Output results
    print(f'KNN. Mean Squared Error: {mse}')
    print(f'KNN. R²: {r2}')
    print(f'KNN. Mean Absolute Error: {mae}')

    return mse , r2, mae

# predict_dose()