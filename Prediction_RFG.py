# House Price Prediction Model
# This script implements a machine learning model to predict house prices using the California Housing dataset.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Loading dataset
def load_data():
    """
    Load the California Housing dataset.
    Returns:
        X (DataFrame): Features
        y (Series): Target variable (house prices)
    """
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='MedHouseVal')
    return X, y

# Preprocessing of data
def preprocess_data(X, y):
    """
    Preprocess the dataset: handle missing values, scale features, and split into train/test sets.
    Args:
        X (DataFrame): Features
        y (Series): Target variable
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler: Split and scaled data
    """
    if X.isnull().sum().any():
        X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train the model
def train_model(X_train, y_train):
    """
    Train a Random Forest Regressor model.
    Args:
        X_train: Scaled training features
        y_train: Training target
    Returns:
        model: Trained RandomForestRegressor
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluating the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using RMSE and R² score.
    Args:
        model: Trained model
        X_test: Scaled test features
        y_test: Test target
    Returns:
        rmse: Root Mean Squared Error
        r2: R² score
        y_pred: Predicted values
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2, y_pred

# Visualizing results of model
def plot_results(y_test, y_pred):
    """
    Plot actual vs predicted house prices.
    Args:
        y_test: Actual test target values
        y_pred: Predicted target values
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual House Prices')
    plt.ylabel('Predicted House Prices')
    plt.title('Actual vs Predicted House Prices')
    plt.tight_layout()
    plt.show()

# Main function for pipeline
def main():
    X, y = load_data()
    print("Dataset loaded. Features:", X.columns.tolist())

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(X, y)
    print("Data preprocessed and split.")

    model = train_model(X_train_scaled, y_train)
    print("Model trained.")

    rmse, r2, y_pred = evaluate_model(model, X_test_scaled, y_test)
    print(f"Model Performance: RMSE = {rmse:.2f}, R² = {r2:.2f}")

    plot_results(y_test, y_pred)
    print("Results visualized.")

    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
