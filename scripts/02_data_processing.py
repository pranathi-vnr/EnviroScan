import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_and_engineer_features():
    logging.info("Data Processing and Feature Engineering")
 
    try:
        air_df = pd.read_csv(config.AIR_QUALITY_FILE)
        weather_df = pd.read_csv(config.WEATHER_FILE)
        osm_df = pd.read_csv(config.OSM_FEATURES_FILE)
    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Run 01_data_collection.py first.")
        return

    df = pd.merge(air_df, weather_df, on=['name', 'latitude', 'longitude'], how='left', suffixes=('_air', '_weather'))
    df = pd.merge(df, osm_df, on=['name', 'latitude', 'longitude'], how='left')
    df['timestamp'] = pd.to_datetime(df['timestamp_air'].fillna(df['timestamp_weather']), unit='s')
    df.drop(columns=[col for col in df if 'timestamp_' in col], inplace=True)
    
    for col in ['temperature', 'humidity', 'wind_speed']:
        df[col].fillna(df[col].median(), inplace=True)
    
    for col in [c for c in df.columns if '_count' in c]:
        df[col].fillna(0, inplace=True)
    
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df.to_csv(config.PROCESSED_EDA_FILE, index=False)
    logging.info(f"Human-readable data saved to {config.PROCESSED_EDA_FILE}")

    df_ml = df.copy()
    available_features = [col for col in config.FEATURE_COLS if col in df_ml.columns]
    
    scaler = MinMaxScaler()
    df_ml[available_features] = scaler.fit_transform(df_ml[available_features])
    logging.info("Numeric features normalized for ML")
    
    joblib.dump(scaler, config.SCALER_FILE)
    logging.info(f"Scaler saved to {config.SCALER_FILE}")
    
    df_ml.to_csv(config.DATA_FOR_ML_FILE, index=False)
    logging.info(f"ML-ready data saved to {config.DATA_FOR_ML_FILE}")
    logging.info("Processing completed successfully")

if __name__ == "__main__":
    process_and_engineer_features()