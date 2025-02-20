import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer

DATA_DIR = "data"
RESULTS_DIR = "results"
LAG_STEPS = 4
WINDOW_SIZE = 12

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_preprocess():
    """Загрузка и предобработка данных"""
    dfs = []

    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv'):
            patient_id = file.split('.')[0]
            df = pd.read_csv(os.path.join(DATA_DIR, file),
                             delimiter=';',
                             parse_dates=['time'])

            df = df.sort_values('time')
            df['hour'] = df['time'].dt.hour
            df['minute'] = df['time'].dt.minute

            df['time_sin'] = np.sin(2 * np.pi * (df['hour'] * 60 + df['minute']) / (24 * 60))
            df['time_cos'] = np.cos(2 * np.pi * (df['hour'] * 60 + df['minute']) / (24 * 60))

            for lag in range(1, LAG_STEPS + 1):
                df[f'glucose_lag_{lag}'] = df['glucose'].shift(lag)
                df[f'carbs_lag_{lag}'] = df['carb_input'].shift(lag)
                df[f'basal_rate_lag_{lag}'] = df['basal_rate'].shift(lag)

            df['glucose_rolling'] = df['glucose'].rolling(WINDOW_SIZE).mean()
            df['carbs_rolling'] = df['carb_input'].rolling(WINDOW_SIZE).sum()
            df['steps_rolling'] = df['steps'].rolling(WINDOW_SIZE).sum()
            df['heart_rate_rolling'] = df['heart_rate'].rolling(WINDOW_SIZE).mean()

            df['patient_id'] = patient_id
            dfs.append(df.dropna())

    return pd.concat(dfs)


def prepare_data(df):
    """Подготовка финального датасета"""
    features = [
        'glucose', 'carb_input', 'heart_rate', 'steps', 'time_sin', 'time_cos', 'bolus_volume_delivered',
    ]

    features += [f'glucose_lag_{i}' for i in range(1, LAG_STEPS + 1)]
    features += [f'carbs_lag_{i}' for i in range(1, LAG_STEPS + 1)]
    features += [f'basal_rate_lag_{i}' for i in range(1, LAG_STEPS + 1)]

    features += ['glucose_rolling', 'carbs_rolling', 'steps_rolling', 'heart_rate_rolling']

    target = 'basal_rate'

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features)
        ],
        remainder='passthrough'
    )

    X = preprocessor.fit_transform(df[features])
    y = df[target].values

    return X, y, preprocessor


def evaluate_models(X, y):
    """Обучение и оценка всех моделей с настройкой гиперпараметров"""
    models = {
        "Linear Regression": LinearRegression(),
        "SVM": SVR(kernel='rbf', C=10.0, epsilon=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=7, weights='distance'),
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10)
    }

    tscv = TimeSeriesSplit(n_splits=3)
    results = {name: {'MSE': [], 'R2': []} for name in models}

    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
    }

    rf_grid_search = RandomizedSearchCV(RandomForestRegressor(), rf_param_grid, n_iter=10, cv=3, random_state=42)

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, model in models.items():

            if name == "Random Forest":
                model = rf_grid_search
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred = np.maximum(y_pred, 0)

            results[name]['MSE'].append(mean_squared_error(y_test, y_pred))
            results[name]['R2'].append(r2_score(y_test, y_pred))

    final_results = {}
    for name in models:
        final_results[name] = {
            'MSE': np.mean(results[name]['MSE']),
            'R2': np.mean(results[name]['R2'])
        }

    return final_results


def plot_metrics(results, patient_id):
    """Визуализация метрик"""
    df = pd.DataFrame(results).T.reset_index()
    df.columns = ['Model', 'MSE', 'R2']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    sns.barplot(x='Model', y='MSE', data=df, ax=ax1, palette='Blues_d')
    ax1.set_title(f'MSE Comparison for Patient {patient_id}')
    ax1.set_ylabel('Mean Squared Error')
    ax1.tick_params(axis='x', rotation=45)

    sns.barplot(x='Model', y='R2', data=df, ax=ax2, palette='Greens_d')
    ax2.set_title(f'R² Score Comparison for Patient {patient_id}')
    ax2.set_ylabel('R² Score')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/metrics_{patient_id}.png")
    plt.close()


def main():
    full_data = load_and_preprocess()
    global_results = []

    for pid in full_data['patient_id'].unique():
        patient_data = full_data[full_data['patient_id'] == pid]

        if len(patient_data) < 100:
            continue

        X, y, _ = prepare_data(patient_data)
        results = evaluate_models(X, y)

        for model_name, metrics in results.items():
            global_results.append({
                'Patient': pid,
                'Model': model_name,
                'MSE': metrics['MSE'],
                'R2': metrics['R2']
            })

    results_df = pd.DataFrame(global_results)

    agg_results = results_df.groupby('Model').agg({
        'MSE': 'mean',
        'R2': 'mean'
    }).round(6).reset_index()

    print("\nFinal Results:")
    print(agg_results[['Model', 'MSE', 'R2']].to_string(index=False))

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.barplot(
        x='Model',
        y='MSE',
        hue='Model',
        data=agg_results,
        palette='Blues',
        legend=False
    )
    plt.title('Average MSE by Model')
    plt.xticks(rotation=45)
    plt.ylabel('MSE')

    plt.subplot(1, 2, 2)
    sns.barplot(
        x='Model',
        y='R2',
        hue='Model',
        data=agg_results,
        palette='Greens',
        legend=False
    )
    plt.title('Average R² Score by Model')
    plt.xticks(rotation=45)
    plt.ylabel('R² Score')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/final_results.png")
    plt.close()


if __name__ == "__main__":
    main()
