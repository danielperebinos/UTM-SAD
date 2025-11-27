import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

LOG_DIR = Path("/app/logs")
CSV_PATH = LOG_DIR / "data_log.csv"
WEIGHTS_PATH = LOG_DIR / "weights.json"


st.set_page_config(page_title="Real-Time Churn Dashboard", layout="wide")
st.title("Real-Time Churn Prediction Dashboard")


def load_logs() -> pd.DataFrame:
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(CSV_PATH)
    except Exception:
        return pd.DataFrame()


def load_weights() -> dict:
    if not WEIGHTS_PATH.exists() or WEIGHTS_PATH.stat().st_size == 0:
        return {}
    try:
        with WEIGHTS_PATH.open() as f:
            return json.load(f)
    except Exception:
        return {}


def render_dashboard() -> None:
    placeholder = st.empty()
    while True:
        with placeholder.container():
            df = load_logs()
            weights = load_weights()

            if df.empty:
                st.info("Waiting for data... The consumer will write logs as soon as messages arrive.")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                total = len(df)
                current_acc = df["current_accuracy"].astype(float).iloc[-1] if "current_accuracy" in df.columns else 0.0
                last_prob = df["probability"].astype(float).iloc[-1] if "probability" in df.columns else 0.0

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Processed", f"{total}")
                col2.metric("Current Accuracy", f"{current_acc:.2%}")
                col3.metric("Last Churn Probability", f"{last_prob:.2%}")

                st.subheader("Model Accuracy Over Time")
                st.line_chart(df.set_index("timestamp")["current_accuracy"].astype(float))

                st.subheader("Real-Time Feature Weights")
                if weights:
                    sorted_items = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)
                    weight_df = pd.DataFrame(sorted_items, columns=["feature", "weight"]).set_index("feature")
                    st.bar_chart(weight_df)
                else:
                    st.info("No weights yet. The consumer will write weights.json after first message.")

        time.sleep(2)


render_dashboard()
