# backend/train.py

import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from app.services.ml_forecasting.feature_eng import clean_and_prepare_data, extract_features

# Robust absolute pathing based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")
WEIGHTS_DIR = os.path.join(BASE_DIR, "app", "services", "ml_forecasting", "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "xgb_duration_model.json")

def run_training():
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Raw data file not found at {RAW_DATA_PATH}")
        return
    
    print("Cleaning and extracting features...")
    cleaned_df = clean_and_prepare_data(df)
    ml_df = extract_features(cleaned_df)
    
    features = ['latitude', 'longitude', 'hour', 'day_of_week', 
                'is_weekend', 'is_rush_hour', 'priority_score', 'cause_severity']
    
    X = ml_df[features]
    y = ml_df['duration_minutes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6, 
        objective='reg:squarederror', 
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Validation Metrics
    print("Evaluating model performance...")
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    
    print("-" * 30)
    print(f"📊 Mean Absolute Error (MAE): {mae:.2f} minutes")
    print(f"📊 Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
    print("-" * 30)
    
    # Save the model in Native JSON format for production
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    # Extract the booster and save
    booster = model.get_booster()
    booster.save_model(MODEL_PATH)
    
    print(f"✅ Training Complete! Model safely written to: {MODEL_PATH}")

if __name__ == "__main__":
    run_training()