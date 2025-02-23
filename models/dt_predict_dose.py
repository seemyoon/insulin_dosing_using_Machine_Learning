from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_preparation.data_preparation import load_and_prepare_data
from pathlib import Path

def predict_dose():
    # Set path to folder with data
    folder_path = Path('data')

    # Prepare data. From data_preparation.py
    X_scaled, y = load_and_prepare_data(folder_path)

    # Split into training and test samples
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # DT model creation and training
    model = DecisionTreeRegressor(max_depth=22, min_samples_leaf=2, min_samples_split=2, random_state=42)
    model.fit(X_train, y_train)

    # Prediction and model evaluation
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Output results
    print(f'Decision Tree. Mean Squared Error: {mse}')
    print(f'Decision Tree. R²: {r2}')
    print(f'Decision Tree. Mean Absolute Error: {mae}')

    return mse , r2, mae

# predict_dose()