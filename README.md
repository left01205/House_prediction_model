# House Price Prediction Project
## Overview
This project implements a machine learning model to predict house prices using the California Housing dataset from scikit-learn. The model employs a Random Forest Regressor to estimate median house values based on features such as median income, house age, and average rooms. The pipeline includes data loading, preprocessing, model training, evaluation, and visualization of results.

## Features

**Dataset** : California Housing dataset with 20,640 samples and 8 numerical features.
**Model** : Random Forest Regressor with 100 trees.
**Preprocessing** : Handles missing values, scales features, and splits data into training (80%) and testing (20%) sets.
**Evaluation Metrics** : Root Mean Squared Error (RMSE) and R² score.
**Visualizations** : Scatter plot of actual vs predicted prices and bar plot of feature importance.

## Requirements

- Python 3.8 or higher
- Required libraries:
 - `pandas`
 - `numpy`
 - `scikit-learn`
 - `matplotlib`
 - `seaborn`



## Install dependencies using:
pip install pandas numpy scikit-learn matplotlib seaborn


## Output

**Console Output** : Displays loaded features, preprocessing status, model training status, and performance metrics (e.g., RMSE = 0.45, R² = 0.85).
**Visualizations** :
Scatter plot of actual vs predicted house prices.
Bar plot of feature importance, highlighting key predictors like median income.



## Performance

- RMSE: Approximately 0.45–0.50 (lower is better).
- R² Score: Approximately 0.80–0.85 (higher is better, max 1.0).
Results may vary slightly due to random data splitting.

## Limitations

The model is trained on the California Housing dataset, which may not generalize to other regions.
Assumes linear relationships in some preprocessing steps, potentially missing complex patterns.
No hyperparameter tuning is performed, which could improve performance.

## Future Improvements

- Add hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Experiment with other models like XGBoost or Neural Networks.
- Incorporate feature engineering (e.g., polynomial features, interaction terms).
- Implement outlier detection and removal for better robustness.


## License
This project is licensed under the MIT License.

