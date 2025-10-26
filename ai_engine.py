import xgboost as xgb
import pandas as pd
import os

def load_model(path="xgboost_model.json"):
    model = xgb.XGBClassifier()
    if os.path.exists(path):
        model.load_model(path)
    else:
        raise FileNotFoundError("Model file not found. Please train it first.")
    return model

def predict_next(data, model):
    features = pd.DataFrame([data['features']])
    prob = model.predict_proba(features)[0][1]
    return prob > 0.7, prob * 100
