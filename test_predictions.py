"""
Test script for validating the multiplier predictor.
Run this after training to verify model behavior.
"""
import pandas as pd
import numpy as np
from ai_engine import load_model, predict_next
import json

def load_feature_ranges():
    """Load valid feature ranges from training"""
    try:
        with open('feature_ranges.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ Warning: feature_ranges.json not found. Run auto_train.py first.")
        return None

def test_prediction(features, model, ranges=None):
    """Test a single prediction case"""
    if ranges:
        # Check if features are within training ranges
        for feat, value in features.items():
            feat_range = ranges[feat]
            if value < feat_range['min'] or value > feat_range['max']:
                print(f"⚠️ Warning: {feat}={value} is outside training range "
                      f"[{feat_range['min']:.2f}, {feat_range['max']:.2f}]")
    
    # Make prediction
    data = {'features': features}
    pred, conf = predict_next(data, model)
    print(f"\nInput features: {features}")
    print(f"Prediction: {'> 2x' if pred else '≤ 2x'}")
    print(f"Confidence: {conf:.1f}%")
    return pred, conf

def main():
    # Load model and feature ranges
    model = load_model()
    ranges = load_feature_ranges()
    
    # Test cases covering different scenarios
    test_cases = [
        # Current live case
        {'avg_5': 2.15, 'avg_10': 2.0, 'std_10': 0.47, 'low_streak': 2},
        
        # Edge cases - near range boundaries
        {'avg_5': 1.99, 'avg_10': 1.93, 'std_10': 0.27, 'low_streak': 0},
        {'avg_5': 2.33, 'avg_10': 2.39, 'std_10': 0.58, 'low_streak': 5},
        
        # Out of training range cases
        {'avg_5': 1.5, 'avg_10': 1.8, 'std_10': 0.2, 'low_streak': 3},
        {'avg_5': 2.8, 'avg_10': 2.5, 'std_10': 0.6, 'low_streak': 0},
        
        # Stable vs volatile cases
        {'avg_5': 2.0, 'avg_10': 2.0, 'std_10': 0.1, 'low_streak': 1},
        {'avg_5': 2.0, 'avg_10': 2.0, 'std_10': 0.8, 'low_streak': 0},
    ]
    
    print("🧪 Running prediction tests...\n")
    for case in test_cases:
        test_prediction(case, model, ranges)
        print("-" * 60)

if __name__ == "__main__":
    main()