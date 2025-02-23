import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from models.cb_predict_dose import predict_dose as cb_predict_dose
from models.dt_predict_dose import predict_dose as dt_predict_dose
from models.knn_predict_dose import predict_dose as knn_predict_dose
from models.lr_predict_dose import predict_dose as lr_predict_dose
from models.rf_predict_dose import predict_dose as rf_predict_dose
from models.xgb_predict_dose import predict_dose as xgb_predict_dose


def get_model_metrics(predict_function):
    """
    Extract MSE, R², and MAE from a model's predict_dose function.
    """
    return predict_function()


def plot_comparison():
    models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'KNN', 'CatBoost', 'XGBoost']
    predict_functions = [lr_predict_dose, dt_predict_dose, rf_predict_dose, knn_predict_dose, cb_predict_dose, xgb_predict_dose]

    # Get all metrics at once for each model
    metrics = [get_model_metrics(func) for func in predict_functions]

    # Unpack the metrics
    mse_results, r2_results, mae_results = zip(*metrics)

    # Create a DataFrame for easy display
    metrics_df = pd.DataFrame({
        'Model': models,
        'MSE': mse_results,
        'R²': r2_results,
        'MAE': mae_results
    })

    # Create a folder to save to if it doesn't exist
    save_folder = 'plot_metrics'
    os.makedirs(save_folder, exist_ok=True)

    # Stacked vertically for better readability
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plot MSE
    sns.barplot(x='Model', y='MSE', data=metrics_df, ax=axes[0])
    axes[0].set_title('Mean Squared Error')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot R²
    sns.barplot(x='Model', y='R²', data=metrics_df, ax=axes[1])
    axes[1].set_title('R²')
    axes[1].tick_params(axis='x', rotation=45)

    # Plot MAE
    sns.barplot(x='Model', y='MAE', data=metrics_df, ax=axes[2])
    axes[2].set_title('Mean Absolute Error')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save the plot
    file_path = os.path.join(save_folder, 'models_comparison.png')
    plt.savefig(file_path)

    plt.close()

    print(f'Graphs have been saved to {file_path}')


plot_comparison()
