import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "delay_model.joblib")

def generate_synthetic_training_data(n_samples: int = 2500) -> pd.DataFrame:
    """
    Generates domain-realistic European cross-border parcel telemetry.
    Features:
      - carrier: Shipping provider
      - last_hub: Current logistics transit facility
      - dwell_time_hours: Time parcel has spent at current checkpoint
      - is_cross_border: 1 if origin country != destination country
    Target:
      - is_delayed: Binary flag (1 = SLA delivery breached, 0 = on schedule)
    """
    np.random.seed(42)

    carriers = ["DHL Express", "DPD Europe", "Hermes Germany", "GLS Logistics", "PostNL", "Colissimo"]
    hubs = [
        "Potsdam Sorting Facility", 
        "Leipzig Hub", 
        "Frankfurt Gateway", 
        "Roissy CDG Airport Customs", 
        "Berlin South Center",
        "Amsterdam Parcel Center"
    ]

    carrier_col = np.random.choice(carriers, n_samples)
    hub_col = np.random.choice(hubs, n_samples)
    dwell_time = np.random.exponential(scale=26.0, size=n_samples)  # Right-skewed realistic wait times
    cross_border = np.random.choice([0, 1], p=[0.35, 0.65], size=n_samples)

    # Domain probability function to simulate real ground truth
    risk_score = (
        (dwell_time > 36.0).astype(float) * 0.45 +
        (hub_col == "Roissy CDG Airport Customs").astype(float) * 0.35 +
        (cross_border == 1).astype(float) * 0.15 +
        np.random.normal(0, 0.05, n_samples)
    )

    is_delayed = (risk_score > 0.42).astype(int)

    return pd.DataFrame({
        "carrier": carrier_col,
        "last_hub": hub_col,
        "dwell_time_hours": dwell_time,
        "is_cross_border": cross_border,
        "is_delayed": is_delayed
    })

def train_and_persist_model():
    """Trains a Random Forest classifier pipeline and saves it to disk."""
    df = generate_synthetic_training_data()
    X = df.drop(columns=["is_delayed"])
    y = df["is_delayed"]

    categorical_cols = ["carrier", "last_hub"]
    numerical_cols = ["dwell_time_hours", "is_cross_border"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

# Load or auto-train on startup
if not os.path.exists(MODEL_PATH):
    ml_pipeline = train_and_persist_model()
else:
    ml_pipeline = joblib.load(MODEL_PATH)

def risk_tier(prob_delayed: float) -> str:
    """Maps a delay probability (0-1) to an operational risk tier."""
    if prob_delayed >= 0.70:
        return "CRITICAL_RISK"
    if prob_delayed >= 0.40:
        return "MODERATE_RISK"
    return "LOW_RISK"

def predict_delay_risk(carrier: str, last_hub: str, dwell_time_hours: float, is_cross_border: int = 1) -> dict:
    """Infers delay probability using the serialized scikit-learn model."""
    input_df = pd.DataFrame([{
        "carrier": carrier,
        "last_hub": last_hub,
        "dwell_time_hours": float(dwell_time_hours),
        "is_cross_border": int(is_cross_border)
    }])

    # Calculate probability of delay (Class 1)
    prob_delayed = float(ml_pipeline.predict_proba(input_df)[0][1])

    return {
        "delay_probability": round(prob_delayed * 100, 1),
        "risk_tier": risk_tier(prob_delayed),
        "model_used": "RandomForest (scikit-learn)"
    }