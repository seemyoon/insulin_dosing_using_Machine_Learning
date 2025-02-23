# Insulin Dose Prediction

## Description
Developing a machine learning model to predict insulin dose based on patient data such as glucose levels, calories, heart rate, steps, and other parameters for type 1 diabetes.

## Setup
To run a specific model, execute the following command in the main directory:

### Random Forest:
```bash
python -m models.rf_predict_dose
```

### CatBoost:
```bash
python -m models.cb_predict_dose
```

### Decision Tree:
```bash
python -m models.dt_predict_dose
```

### K-Nearest Neighbors:
```bash
python -m models.knn_predict_dose
```

### Linear Regression:
```bash
python -m models.lr_predict_dose
```

### Support Vector Machine:
```bash
python -m models.svm_predict_dose
```

### XGBoost:
```bash
python -m models.xgb_predict_dose
```

### All models at once:
```bash
python -m plot_metrics.plot_metrics
```


## Dataset
Hidalgo, J. Ignacio; Alvarado, Jorge; Botella, Marta; Aramendi, Aranzazu; Velasco, J. Manuel; Garnica, Oscar (2024), “HUPA-UCM Diabetes Dataset”, Mendeley Data, V1, doi: [10.17632/3hbcscwz44.1](https://doi.org/10.17632/3hbcscwz44.1)