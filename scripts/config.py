import os
from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(dotenv_path=PROJECT_ROOT / '.env')

API_KEY = os.getenv("OPENWEATHER_API_KEY")

OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

LOCATIONS_FILE = DATA_DIR / "locations.csv"

AIR_QUALITY_FILE = OUTPUT_DIR / "raw_air_quality.csv"
WEATHER_FILE = OUTPUT_DIR / "raw_weather.csv"
OSM_FEATURES_FILE = OUTPUT_DIR / "raw_osm_features.csv"

PROCESSED_EDA_FILE = OUTPUT_DIR / "processed_data_for_eda.csv"
DATA_FOR_ML_FILE = OUTPUT_DIR / "processed_data_for_ml.csv"
SCALER_FILE = OUTPUT_DIR / "scaler.joblib"

LABELED_FILE = OUTPUT_DIR / "labeled_data_for_dashboard.csv"
TRAIN_FILE = OUTPUT_DIR / "train_data.csv"
TEST_FILE = OUTPUT_DIR / "test_data.csv"

MODEL_FILE = OUTPUT_DIR / "pollution_source_model.joblib"
ENCODER_FILE = OUTPUT_DIR / "label_encoder.joblib"
EVALUATION_FILE = OUTPUT_DIR / "model_evaluation_report.txt"
CONFUSION_MATRIX_FILE = OUTPUT_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.png"

FEATURE_COLS = [
    'hour', 'day_of_week', 'month', 'temperature', 'humidity', 'wind_speed',
    'roads_count', 'industrial_count', 'agriculture_count', 'dumps_count',
    'co', 'nh3', 'no', 'no2', 'o3', 'pm10', 'pm2_5', 'so2'
]
TARGET_COL = 'pollution_source'
TEST_SIZE = 0.2
RANDOM_STATE = 42