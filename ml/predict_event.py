import json
import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "lightgbm_source_classifier.pkl"
FEATURE_PATH = MODEL_DIR / "prototype_features.json"
LABEL_PATH = MODEL_DIR / "label_mapping.json"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, "r", encoding="utf-8") as f:
    FEATURES = json.load(f)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_data = json.load(f)

INT_TO_LABEL = {
    int(k): v
    for k, v in label_data["int_to_label"].items()
}


# ============================================================
# PREDICT EVENT
# ============================================================

def predict_event(event_data):
    """
    Predict the likely thermal source for one event.
    """

    # Convert input dictionary to DataFrame
    X = pd.DataFrame([event_data])

    # Make sure all required features exist
    for feature in FEATURES:

        if feature not in X.columns:
            X[feature] = 0

    # Keep exact training feature order
    X = X[FEATURES]

    # Numeric conversion
    for feature in FEATURES:
        X[feature] = pd.to_numeric(
            X[feature],
            errors="coerce"
        )

    X = X.replace(
        [float("inf"), float("-inf")],
        0
    )

    X = X.fillna(0)

    # ========================================================
    # PREDICTION
    # ========================================================

    probabilities = model.predict_proba(X)[0]

    predicted_index = probabilities.argmax()

    predicted_source = INT_TO_LABEL[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = {
        "predicted_source": predicted_source,
        "confidence": round(confidence, 4),
        "probabilities": {
            INT_TO_LABEL[i]: round(
                float(probabilities[i]),
                4
            )
            for i in range(len(probabilities))
        }
    }

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_event = {
        "observation_count": 16,
        "mean_frp": 2.55,
        "peak_frp": 7.50,
        "total_frp": 40.8,
        "mean_brightness": 310,
        "persistent": 1,
        "historical_detection_count": 114,
        "historical_mean_frp": 2.4,
        "local_frp_deviation": 0.15,
        "historical_daily_activity": 0.5,
        "previously_detected": 1,
        "anomaly_score": -0.31,
        "is_anomaly": 1
    }

    result = predict_event(
        sample_event
    )

    print("\nPrediction:")
    print(
        json.dumps(
            result,
            indent=4
        )
    )
    