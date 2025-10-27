import xgboost as xgb
import pandas as pd
import os

# Explicit feature order used during training — keep this in sync with training code
FEATURE_NAMES = ["avg_5", "avg_10", "std_10", "low_streak"]


def load_model(path="xgboost_model.json"):
    model = xgb.XGBClassifier()
    if os.path.exists(path):
        model.load_model(path)
    else:
        raise FileNotFoundError("Model file not found. Please train it first.")
    return model


def predict_next(data, model):
    # Build DataFrame with the exact feature order expected by the model
    feature_dict = data.get('features', {})
    missing = [f for f in FEATURE_NAMES if f not in feature_dict]
    if missing:
        raise KeyError(f"Missing feature(s) for prediction: {missing}")

    features = pd.DataFrame([feature_dict])[FEATURE_NAMES]

    # Use predict_proba; if the model doesn't support it this will raise AttributeError
    prob = model.predict_proba(features)[0][1]

    # Return boolean decision and confidence as a percentage
    return prob > 0.7, prob * 100
