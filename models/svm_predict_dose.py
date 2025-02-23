from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from data_preparation.data_preparation import load_and_prepare_data


def predict_dose():
    # Set path to folder with data
    folder_path = Path('data')

    # Prepare data. From data_preparation.py
    X_scaled, y = load_and_prepare_data(folder_path)

    # Split into training and test samples
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # SVM model creation and training
    model = SVR(kernel='linear', C=1, gamma='scale') # it takes a long time to train
    model.fit(X_train, y_train)

    # Prediction and model evaluation
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Output results
    print(f'Support Vector Regression. Mean Squared Error: {mse}')
    print(f'Support Vector Regression. R²: {r2}')
    print(f'Support Vector Regression. Mean Absolute Error: {mae}')

    return mse , r2, mae

predict_dose()