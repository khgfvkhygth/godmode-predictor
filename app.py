import streamlit as st
import pandas as pd
from ai_engine import load_model, predict_next
from live_feed import get_latest_data

st.set_page_config(page_title="GodMode AI Predictor", layout="wide")
st.title("?? GodMode v3 – Live AI Crash Predictor")

model = load_model()
latest_data = get_latest_data()

if latest_data:
    prediction, confidence = predict_next(latest_data, model)
    
    st.metric("?? Last Multiplier", f"{latest_data['last_multiplier']}x")
    st.metric("?? AI Prediction", "> 2x" if prediction else "? 2x")
    st.metric("? Confidence", f"{confidence:.1f}%")
    
    if prediction and confidence >= 70:
        st.success("?? Spike likely – High Confidence Zone!")
    elif confidence < 50:
        st.warning("?? Low confidence – Risky zone detected.")

    try:
        df = pd.read_csv("prediction_log.csv")
        st.subheader("?? Recent Predictions")
        st.dataframe(df.tail(10), use_container_width=True)
    except FileNotFoundError:
        st.info("No prediction log yet.")
else:
    st.warning("Waiting for live data...")
