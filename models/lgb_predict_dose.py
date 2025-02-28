from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from data_preparation.data_preparation import load_and_prepare_data

def predict_dose():
    # Prepare data from data_preparation.py
    X, y = load_and_prepare_data()

    # Split into training and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """
    We do standardization after splitting the data into training and test samples. 
    Because in real life, test data is something your model hasn't seen yet.
    They have to be brand new to correctly assess the quality of the model.
    """
    # Data scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LightGBM model creation and training
    model = lgb.LGBMRegressor(n_estimators=800, subsample=1.0, reg_lambda=1, reg_alpha=0.1, max_depth=14, learning_rate=0.05,
                              num_leaves=31, colsample_bytree=0.6)
    model.fit(X_train_scaled, y_train)

    # Prediction and model evaluation
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    mse_rounded = round(mse, 6)
    r2_rounded = round(r2, 6)
    mae_rounded = round(mae, 6)

    # Output results
    print(f'LightGBM. Mean Squared Error: {mse_rounded}')
    print(f'LightGBM. R²: {r2_rounded}')
    print(f'LightGBM. Mean Absolute Error: {mae_rounded}')

    return mse, r2, mae

# predict_dose()
