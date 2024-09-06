import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def load_data():
    # Load datasets
    race_data = pd.read_csv('../data/race_data.csv')
    telemetry_data = pd.read_csv('../data/telemetry_data.csv')
    
    # Optionally, you might load data from an API here as well
    # Example: race_results = get_race_results(2022, 1) from Ergast API

    return race_data, telemetry_data

def clean_data(race_data, telemetry_data):
    # Handle missing values (example: forward fill)
    race_data.fillna(method='ffill', inplace=True)
    telemetry_data.fillna(method='bfill', inplace=True)
    
    # Normalize telemetry data (e.g., speed, throttle, brake)
    scaler = MinMaxScaler()
    telemetry_data[['speed', 'throttle', 'brake']] = scaler.fit_transform(telemetry_data[['speed', 'throttle', 'brake']])

    return race_data, telemetry_data

def engineer_features(race_data, telemetry_data):
    # Example: speed differential between two drivers
    telemetry_data['speed_diff'] = telemetry_data['driver1_speed'] - telemetry_data['driver2_speed']
    
    # Example: create a feature for DRS zone
    def is_drs_zone(track_position):
        drs_zones = [(1000, 1200), (3000, 3200)]  # Example DRS zone positions
        return any(zone[0] <= track_position <= zone[1] for zone in drs_zones)

    telemetry_data['is_drs_zone'] = telemetry_data['track_position'].apply(is_drs_zone)

    # More feature engineering (e.g., tire compounds, lap time differences)
    # Example: one-hot encode categorical variables
    encoder = OneHotEncoder()
    tire_encoded = encoder.fit_transform(race_data[['tyre_compound']]).toarray()
    tire_encoded_df = pd.DataFrame(tire_encoded, columns=encoder.get_feature_names_out(['tyre_compound']))
    race_data = pd.concat([race_data, tire_encoded_df], axis=1)
    
    return race_data, telemetry_data

def save_data(race_data, telemetry_data):
    # Save processed data for later use
    race_data.to_csv('../data/processed_race_data.csv', index=False)
    telemetry_data.to_csv('../data/processed_telemetry_data.csv', index=False)

if __name__ == '__main__':
    # Load data
    race_data, telemetry_data = load_data()
    
    # Clean data
    race_data, telemetry_data = clean_data(race_data, telemetry_data)
    
    # Feature engineering
    race_data, telemetry_data = engineer_features(race_data, telemetry_data)
    
    # Save processed data
    save_data(race_data, telemetry_data)
