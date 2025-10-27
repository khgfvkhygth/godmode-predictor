import pandas as pd
import sys
sys.path.append(r"c:\Users\bill\Desktop\godmode-predictor\godmode-predictor")
from ai_engine import load_model, predict_next
from live_feed import get_latest_data

# Load and peek at training data
df = pd.read_csv("multiplier_log.csv", encoding='utf-8')
print("Training data shape:", df.shape)
print("\nFeature columns:", list(df.columns))
print("\nSample rows:")
print(df.head(3).to_string())

# Load model and run prediction
model = load_model('xgboost_model.json')
test_data = get_latest_data()
pred, conf = predict_next(test_data, model)
print("\nPrediction test:")
print("Input features:", test_data['features'])
print("Prediction:", pred)
print("Confidence:", conf)