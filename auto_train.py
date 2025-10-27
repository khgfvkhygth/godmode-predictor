import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

def retrain_model(input_csv="multiplier_log.csv"):
    # Load and prepare data
    df = pd.read_csv(input_csv, encoding='utf-8')
    df['target'] = (df['next_multiplier'] > 2).astype(int)

    # Calculate and save feature ranges for validation
    features = ['avg_5', 'avg_10', 'std_10', 'low_streak']
    feature_ranges = {
        col: {'min': float(df[col].min()), 'max': float(df[col].max())}
        for col in features
    }
    
    with open('feature_ranges.json', 'w') as f:
        json.dump(feature_ranges, f, indent=2)
    
    X = df[features]
    y = df['target']

    # Print class distribution
    print("\nClass distribution in training data:")
    print(y.value_counts(normalize=True))

    # Use stratified split to maintain class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Configure XGBoost with:
    # - max_delta_step to stabilize updates with extreme class imbalance
    # - min_child_weight to require more evidence for splits
    # - subsample to reduce overfitting
    model = xgb.XGBClassifier(
        max_delta_step=1,
        min_child_weight=2,
        subsample=0.8,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    # Train with validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )

    # Evaluate model
    y_pred = model.predict(X_test)
    print("\nModel evaluation on test set:")
    print(classification_report(y_test, y_pred))

    # Save model
    model.save_model("xgboost_model.json")
    print("✅ Model retrained and saved to xgboost_model.json")

    # Test predictions on some example cases
    print("\nExample predictions:")
    test_cases = [
        {'avg_5': 2.15, 'avg_10': 2.0, 'std_10': 0.47, 'low_streak': 2},
        {'avg_5': 1.5, 'avg_10': 1.8, 'std_10': 0.2, 'low_streak': 3},
        {'avg_5': 2.8, 'avg_10': 2.5, 'std_10': 0.6, 'low_streak': 0}
    ]
    
    for case in test_cases:
        features = pd.DataFrame([case])
        prob = model.predict_proba(features)[0][1]
        print(f"\nInput: {case}")
        print(f"Probability of >2x: {prob:.1%}")

if __name__ == "__main__":
    retrain_model()

