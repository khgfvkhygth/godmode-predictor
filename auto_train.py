import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

def retrain_model(input_csv="multiplier_log.csv"):
    df = pd.read_csv(input_csv)
    df['target'] = (df['next_multiplier'] > 2).astype(int)

    features = ['avg_5', 'avg_10', 'std_10', 'low_streak']
    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    model.save_model("xgboost_model.json")
    print("? Model retrained and saved to xgboost_model.json")

if __name__ == "__main__":
    retrain_model()

