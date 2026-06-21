# backend/app/services/ml_forecasting/feature_eng.py

import pandas as pd
import numpy as np
from typing import Dict, Any

# Define weights for the Impact Score Formula: I = alpha*S(E) + beta*D(T,L) + gamma*T_pred
ALPHA = 0.4  # Weight for Event Severity
BETA = 0.3   # Weight for Traffic Density (Proxy)
GAMMA = 0.3  # Weight for Predicted Duration

# Severity Mapping S(E) out of 100
SEVERITY_MAP = {
    'vip movement': 95,
    'procession': 90,
    'protest': 85,
    'water_logging': 80,
    'tree_fall': 75,
    'accident': 70,
    'construction': 60,
    'vehicle_breakdown': 50,
    'pot_holes': 40,
    'others': 30
}

def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ruthless data cleaning: Handles datetimes, calculates duration, and removes extreme outliers.
    """
    data = df.copy()
    
    # 1. Parse Datetimes safely
    data['start_datetime'] = pd.to_datetime(data['start_datetime'], errors='coerce', utc=True)
    data['resolved_datetime'] = pd.to_datetime(data['resolved_datetime'], errors='coerce', utc=True)
    data['closed_datetime'] = pd.to_datetime(data['closed_datetime'], errors='coerce', utc=True)
    
    # 2. Calculate Ground Truth Duration (Minutes)
    # Prefer resolved_datetime, fallback to closed_datetime
    end_time = data['resolved_datetime'].fillna(data['closed_datetime'])
    data['duration_minutes'] = (end_time - data['start_datetime']).dt.total_seconds() / 60.0
    
    # 3. Outlier Removal (The most critical step)
    # Drop where start or end time is missing
    data = data.dropna(subset=['start_datetime', 'duration_minutes'])
    
    # Drop negative durations (logging errors)
    data = data[data['duration_minutes'] > 0]
    
    # Drop extreme outliers using the Interquartile Range (IQR) method
    Q1 = data['duration_minutes'].quantile(0.25)
    Q3 = data['duration_minutes'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 3 * IQR  # Using 3x IQR for a slightly more forgiving bound on traffic events
    
    # Cap maximum duration to a realistic 24 hours (1440 minutes) to avoid model skew
    max_allowed = min(upper_bound, 1440)
    data = data[data['duration_minutes'] <= max_allowed]
    
    return data

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts spatial and temporal features for the ML Model.
    """
    data = df.copy()
    
    # Temporal Features
    data['hour'] = data['start_datetime'].dt.hour
    data['day_of_week'] = data['start_datetime'].dt.dayofweek
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    
    # Rush Hour Proxy (Morning 8-11, Evening 17-20)
    rush_morning = (data['hour'] >= 8) & (data['hour'] <= 11)
    rush_evening = (data['hour'] >= 17) & (data['hour'] <= 20)
    data['is_rush_hour'] = (rush_morning | rush_evening).astype(int)
    
    # Categorical Encoding (Simple mapping for tree-based models)
    # Map priorities to ordinal values
    priority_map = {'high': 3, 'medium': 2, 'low': 1}
    data['priority'] = data['priority'].astype(str).str.strip().str.lower()
    data['priority_score'] = data['priority'].map(priority_map).fillna(1)
    
    # Target Encoding for Event Cause (Mapping string to severity integer)
    data['event_cause'] = data['event_cause'].astype(str).str.strip().str.lower()
    data['cause_severity'] = data['event_cause'].map(SEVERITY_MAP).fillna(30)
    
    # Drop unnecessary text/ID columns for ML input
    features = ['latitude', 'longitude', 'hour', 'day_of_week', 
                'is_weekend', 'is_rush_hour', 'priority_score', 'cause_severity']
    
    return data[features + ['duration_minutes']]

def calculate_impact_score(event_cause: str, priority: str, hour: int, predicted_duration: float) -> float:
    """
    Calculates the composite Impact Score using the mathematical formula.
    I = α * S(E) + β * D(T,L) + γ * T_pred
    """
    # 1. S(E): Severity of Event (Normalized 0-1)
    event_cause_clean = str(event_cause).strip().lower()
    priority_clean = str(priority).strip().lower()
    s_e = SEVERITY_MAP.get(event_cause_clean, 30) / 100.0
    
    # 2. D(T,L): Density Proxy based on Time and Location (Priority)
    is_rush = 1 if (8 <= hour <= 11) or (17 <= hour <= 20) else 0
    priority_val = {'high': 1.0, 'medium': 0.6, 'low': 0.2}.get(priority_clean, 0.2)
    # Density is highest during rush hour on high priority corridors
    d_tl = (is_rush * 0.5) + (priority_val * 0.5)
    
    # 3. T_pred: Predicted Duration (Normalized against a 4-hour severe benchmark)
    t_pred_norm = min(predicted_duration / 240.0, 1.0) 
    
    # Final Calculation (Scale up to 100 for easy UI reading)
    impact_score = (ALPHA * s_e) + (BETA * d_tl) + (GAMMA * t_pred_norm)
    
    return round(impact_score * 100, 2)