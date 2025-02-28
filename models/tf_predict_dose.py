import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_preparation.data_preparation import load_and_prepare_data


def predict_dose():
    # Prepare data
    X, y = load_and_prepare_data()

    # Split into training and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Data scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Neural network model creation and training
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Prediction
    y_pred = model.predict(X_test_scaled).flatten()

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    mse_rounded = round(mse, 6)
    r2_rounded = round(r2, 6)
    mae_rounded = round(mae, 6)

    # Output results
    print(f'TensorFlow NN. Mean Squared Error: {mse_rounded}')
    print(f'TensorFlow NN. R²: {r2_rounded}')
    print(f'TensorFlow NN. Mean Absolute Error: {mae_rounded}')

    return mse, r2, mae

predict_dose()
