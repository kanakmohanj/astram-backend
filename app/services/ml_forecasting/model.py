# backend/app/services/ml_forecasting/model.py

import xgboost as xgb
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from .feature_eng import clean_and_prepare_data, extract_features

# 1. Use .json for security (no arbitrary code execution), compatibility, and stability
MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "xgb_duration_model.json")

class TrafficForecaster:
    def __init__(self):
        self.booster = None
        self.features = ['latitude', 'longitude', 'hour', 'day_of_week', 
                         'is_weekend', 'is_rush_hour', 'priority_score', 'cause_severity']
        # Load immediately upon server startup to prevent blocking latency
        self._load_model()

    def _load_model(self):
        """Loads the JSON model into memory natively for instantaneous inference."""
        if os.path.exists(MODEL_PATH):
            try:
                self.booster = xgb.Booster()
                self.booster.load_model(MODEL_PATH)
                print("🚀 XGBoost Forecasting Model loaded into RAM successfully.")
            except Exception as e:
                print(f"⚠️ Failed to load model weights: {e}")
        else:
            print(f"⚠️ Model weights not found at {MODEL_PATH}. Training required.")

    def train(self, raw_csv_path: str):
        """
        End-to-end training pipeline. Run this once offline or via an admin endpoint.
        """
        print("Loading data...")
        df = pd.read_csv(raw_csv_path)
        
        print("Cleaning data...")
        cleaned_df = clean_and_prepare_data(df)
        
        print("Extracting features...")
        ml_df = extract_features(cleaned_df)
        
        X = ml_df[self.features]
        y = ml_df['duration_minutes']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training XGBoost Regressor...")
        # We use the Scikit-Learn wrapper for training convenience
        sk_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )
        
        sk_model.fit(X_train, y_train)
        
        # Validation Metrics
        preds = sk_model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        print(f"Model Trained! MAE: {mae:.2f} mins, RMSE: {rmse:.2f} mins")
        
        # Ensure weights directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # 2. Save using native XGBoost JSON format (No joblib/pickle)
        sk_model.save_model(MODEL_PATH)
        print(f"Model safely saved to {MODEL_PATH}")
        
        # Reload the booster into memory for immediate use without restarting the server
        self._load_model()

    def predict_duration(self, raw_event_data: dict) -> float:
        """
        Predicts the duration of a single live event.
        Called by FastAPI endpoints.
        """
        if self.booster is None:
            raise RuntimeError("Model not trained yet! Please train or provide weights.")
            
        try:
            import pandas as pd
            import xgboost as xgb
            
            # LIVE INFERENCE FIX: Skip extract_features() because the live API 
            # payload already contains the engineered features (hour, day_of_week, etc.)
            processed_features = {
                'latitude': float(raw_event_data.get('latitude', 12.9716)),
                'longitude': float(raw_event_data.get('longitude', 77.5946)),
                'hour': int(raw_event_data.get('hour', 12)),
                'day_of_week': int(raw_event_data.get('day_of_week', 0)),
                'is_weekend': int(raw_event_data.get('is_weekend', 0)),
                'is_rush_hour': int(raw_event_data.get('is_rush_hour', 0)),
                'priority_score': int(raw_event_data.get('priority_score', 2)),
                'cause_severity': int(raw_event_data.get('cause_severity', 30))
            }
            
            # 1. Convert to DataFrame
            df_features = pd.DataFrame([processed_features])
            
            # 2. Ensure strict column order matches self.features before converting to DMatrix
            df_input = df_features[self.features]
            
            # 3. Use native DMatrix for maximum inference speed
            dmatrix = xgb.DMatrix(df_input)
            
            prediction = self.booster.predict(dmatrix)
            
            # 4. Clamp negative predictions to 0.0 mathematically
            return max(0.0, float(prediction[0]))
            
        except KeyError as e:
            raise ValueError(f"Missing required features: {e}")
        except Exception as e:
            raise ValueError(f"Inference failed. Check input data formatting: {e}")

# Singleton instance for easy import in FastAPI routes
forecaster_instance = TrafficForecaster()



# # backend/app/services/ml_forecasting/model.py

# import xgboost as xgb
# import pandas as pd
# import joblib
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, root_mean_squared_error
# from .feature_eng import clean_and_prepare_data, extract_features

# MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "xgb_duration_model.pkl")

# class TrafficForecaster:
#     def __init__(self):
#         self.model = None
#         self.features = ['latitude', 'longitude', 'hour', 'day_of_week', 
#                          'is_weekend', 'is_rush_hour', 'priority_score', 'cause_severity']

#     def train(self, raw_csv_path: str):
#         """
#         End-to-end training pipeline. Run this once offline or via an admin endpoint.
#         """
#         print("Loading data...")
#         df = pd.read_csv(raw_csv_path)
        
#         print("Cleaning data...")
#         cleaned_df = clean_and_prepare_data(df)
        
#         print("Extracting features...")
#         ml_df = extract_features(cleaned_df)
        
#         X = ml_df[self.features]
#         y = ml_df['duration_minutes']
        
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         print("Training XGBoost Regressor...")
#         # Hyperparameters tuned for tabular traffic data
#         self.model = xgb.XGBRegressor(
#             n_estimators=200,
#             learning_rate=0.05,
#             max_depth=6,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             objective='reg:squarederror',
#             random_state=42
#         )
        
#         self.model.fit(X_train, y_train)
        
#         # Validation Metrics
#         preds = self.model.predict(X_test)
#         mae = mean_absolute_error(y_test, preds)
#         rmse = root_mean_squared_error(y_test, preds)
#         print(f"Model Trained! MAE: {mae:.2f} mins, RMSE: {rmse:.2f} mins")
        
#         # Ensure weights directory exists
#         os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
#         joblib.dump(self.model, MODEL_PATH)
#         print(f"Model saved to {MODEL_PATH}")

#     def predict_duration(self, event_features: dict) -> float:
#         """
#         Predicts the duration of a single live event.
#         Called by FastAPI endpoints.
#         """
#         if self.model is None:
#             if os.path.exists(MODEL_PATH):
#                 self.model = joblib.load(MODEL_PATH)
#             else:
#                 raise Exception("Model not trained yet! Please train or provide weights.")
                
#         # Convert incoming JSON/dict to DataFrame format required by XGBoost
#         df_input = pd.DataFrame([event_features])
        
#         # Ensure correct column order
#         df_input = df_input[self.features]
        
#         prediction = self.model.predict(df_input)
#         return max(0.0, float(prediction[0]))

# # Singleton instance for easy import in FastAPI routes
# forecaster_instance = TrafficForecaster()
