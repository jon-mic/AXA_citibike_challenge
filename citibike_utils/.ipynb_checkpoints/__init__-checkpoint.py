import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler 
from typing import Dict, List, Tuple

def create_cyclic_time_features(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclic features for hour of the day.
    """
    merged_data['hour_sin'] = np.sin(2 * np.pi * merged_data['hour'] / 24)
    merged_data['hour_cos'] = np.cos(2 * np.pi * merged_data['hour'] / 24)
    return merged_data


def normalize_weather_features(merged_data: pd.DataFrame, scaler=None) -> pd.DataFrame:
    """
    Normalize weather features using StandardScaler and OneHotEncoder.
    """
    # one-hot encode WMO code
    #encoder = OneHotEncoder(drop='first')
    #wmo_encoded = encoder.fit_transform(merged_data[['weather_code']])
    #encoded_column_names = encoder.get_feature_names_out(['weather_code'])
    #wmo_encoded_df = pd.DataFrame(wmo_encoded, columns=encoded_column_names)
    #wmo_encoded_df.index = accidents.index
    #merged_data = pd.concat([merged_data.drop('weather_code', axis=1), weather_encoded_df], axis=1)

    # scale numerical weather features
    weather_features = [
        'temperature_2m_C',
        'relative_humidity_2m_perc',
        'precipitation_mm',
        'wind_speed_10m_km_h',
        'pressure_msl_hPa',
        'cloud_cover_perc',
    ]
    if not scaler:
        scaler = StandardScaler()
        merged_data[weather_features] = scaler.fit_transform(merged_data[weather_features])
    else:
        merged_data[weather_features] = scaler.transform(merged_data[weather_features])

    return merged_data, scaler

def euclidean_distance(predt: np.ndarray, dtrain: xgb.DMatrix):
    """
    Calculate the Euclidean (geometric) distance between true and predicted lat/lon coordinates.
    
    """
    labels = dtrain.get_label().reshape(-1, 2)
    preds = predt.reshape(-1, 2)
    return np.sqrt(np.sum((preds - labels) ** 2))

def haversine_distance(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    """
    Calculate the Haversine distance between true and predicted lat/lon coordinates.
    """
    labels = dtrain.get_label().reshape(-1, 2)
    preds = predt.reshape(-1, 2)

    # Radius of the Earth in meters
    R = 6378137

    lat_true = np.radians(labels[:, 0])
    lon_true = np.radians(labels[:, 1])
    lat_pred = np.radians(preds[:, 0])
    lon_pred = np.radians(preds[:, 1])

    dlat = lat_pred - lat_true
    dlon = lon_pred - lon_true
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_true) * np.cos(lat_pred) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # Distance in meters

    return distance

def mean_haversine_distance(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    return 'mean_haversine_distance', np.mean(haversine_distance(predt, dtrain))

def haversine_loss(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom loss function for XGBoost using Haversine distance.
    Returns the gradient and the hessian for the Haversine distance.
    """
    labels = dtrain.get_label().reshape(-1, 2)  # Reshape true values into [lat, lon]
    predt = predt.reshape(-1, 2)  # Reshape predictions into [lat_pred, lon_pred]

    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert degrees to radians for haversine calculation
    lat_true = np.radians(labels[:, 0])
    lon_true = np.radians(labels[:, 1])
    lat_pred = np.radians(predt[:, 0])
    lon_pred = np.radians(predt[:, 1])

    # Haversine formula components
    dlat = lat_pred - lat_true
    dlon = lon_pred - lon_true
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_true) * np.cos(lat_pred) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Gradients (partial derivatives)
    grad_lat = R * np.sin(dlat) * c  # Derivative wrt latitude
    grad_lon = R * np.sin(dlon) * c  # Derivative wrt longitude

    # Gradient and hessian (second derivative)
    grad = np.hstack((grad_lat[:, None], grad_lon[:, None])).flatten()
    hess = np.ones_like(grad)  # Simplified Hessian (constant)

    return grad, hess

def evaluate_loc_model(booster, X_test, y_test):
    """
    Evaluates the model by calculating the Haversine distance and error metrics.
    """
    # Create DMatrix for the test data
    X_test['start_station_name'] = X_test['start_station_name'].astype('category')
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True, nthread=4)
    
    # Generate predictions
    preds = booster.predict(dtest)
    
    # Calculate Haversine distances
    distances = haversine_distance(preds, dtest)

    # Calculate residuals
    labels = dtest.get_label().reshape(preds.shape)
    residuals = preds - labels

    # Compute metrics
    mean_haversine_distance = np.mean(distances)
    mean_euclidean_distance = np.mean(euclidean_distance(preds, dtest))
    mean_absolute_error = np.mean(np.abs(residuals))
    root_mean_squared_error = np.sqrt(np.mean(residuals ** 2))

    # Print evaluation results
    print(f'Mean Absolute Error: {mean_absolute_error:.2f} grad')
    print(f'Root Mean Squared Error: {root_mean_squared_error:.2f} grad')
    print(f'Mean Euclidean Distance: {mean_euclidean_distance:.2f} grad')
    print(f'Mean Haversine Distance: {mean_haversine_distance:.2f} m')
    print(f'Maximum Predicted Distance: {np.max(distances):.2f} m')
    return pd.DataFrame(distances)