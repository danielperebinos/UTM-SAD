import json
import time
from pathlib import Path

import altair as alt
import pandas as pd
from pandas.io.formats.style import Styler
import streamlit as st

LOG_DIR = Path("/app/logs")
CSV_PATH = LOG_DIR / "data_log.csv"
WEIGHTS_PATH = LOG_DIR / "weights.json"

st.set_page_config(page_title="🏦 Real-Time Churn Detection System", layout="wide")
st.title("🏦 Real-Time Churn Detection System")


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


def clear_logs():
    if CSV_PATH.exists():
        CSV_PATH.unlink()
    if WEIGHTS_PATH.exists():
        WEIGHTS_PATH.unlink()


def color_for_accuracy(acc: float) -> str:
    if acc >= 0.8:
        return "🟢"
    if acc >= 0.5:
        return "🟠"
    return "🔴"


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def styled_recent(df: pd.DataFrame) -> Styler:
    def highlight(row):
        return ["background-color: #ffe5e5" if row["prediction"] != row["true_label"] else "" for _ in row]

    return df.style.apply(highlight, axis=1)


def render_dashboard() -> None:
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 0.5, 5.0, 2.0, 0.5)
    if st.sidebar.button("Clear Logs"):
        clear_logs()
        st.sidebar.success("Logs cleared. Waiting for fresh data...")

    st.sidebar.caption("Use the slider to control how often the dashboard refreshes.")

    placeholder = st.empty()
    while True:
        with placeholder.container():
            df = load_logs()
            weights = load_weights()

            if df.empty:
                st.info("Waiting for data... The consumer will write logs as soon as messages arrive.")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["current_accuracy"] = pd.to_numeric(df["current_accuracy"], errors="coerce")
                df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
                df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
                df["true_label"] = pd.to_numeric(df["true_label"], errors="coerce")

                total = len(df)
                current_acc = df["current_accuracy"].iloc[-1] if not df["current_accuracy"].isna().all() else 0.0
                current_acc = 0.0 if pd.isna(current_acc) else current_acc
                prev_acc = (
                    df["current_accuracy"].iloc[-2]
                    if len(df) > 1 and not df["current_accuracy"].iloc[-2:].isna().any()
                    else current_acc
                )
                delta_acc = (current_acc - prev_acc) * 100 if prev_acc is not None else 0.0
                last_prob = df["probability"].iloc[-1] if not df["probability"].isna().all() else 0.0
                last_prob = 0.0 if pd.isna(last_prob) else last_prob

                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Total Processed", f"{total}", delta=None)
                kpi2.metric(
                    f"Global Accuracy {color_for_accuracy(current_acc)}",
                    f"{current_acc:.2%}",
                    f"{delta_acc:+.2f} pts",
                )
                kpi3.metric("Current Model State", "River - Logistic Regression")

                # Live feed section
                left, right = st.columns(2)
                latest = df.iloc[-1]

                with left:
                    st.subheader("Last Customer Snapshot")
                    snapshot = st.container()
                    with snapshot:
                        correct = latest["prediction"] == latest["true_label"]
                        status_icon = "✅ Correct" if correct else "❌ Incorrect"
                        st.markdown(f"**Prediction vs Truth:** {status_icon}")
                        cols = st.columns(4)
                        cols[0].metric("Customer ID", latest.get("customer_id", latest.get("row_id", "")))
                        cols[1].metric("Age", latest.get("age", ""))
                        cols[2].metric("Gender", latest.get("gender", ""))
                        cols[3].metric("Geography", latest.get("geography", ""))
                        cols2 = st.columns(3)
                        cols2[0].metric("Balance", f"{safe_float(latest.get('balance', 0)):.2f}")
                        cols2[1].metric("Credit Score", latest.get("credit_score", ""))
                        cols2[2].metric("Products", latest.get("num_products", ""))
                        st.write(
                            f"True Label: **{safe_int(latest['true_label'])}** | Prediction: **{safe_int(latest['prediction'])}**"
                        )
                        prob = safe_float(latest.get("probability", 0.0))
                        st.write("Churn Probability")
                        st.progress(min(max(prob, 0.0), 1.0))

                with right:
                    st.subheader("XAI - Feature Weights")
                    if weights:
                        sorted_items = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)
                        weight_df = pd.DataFrame(sorted_items, columns=["feature", "weight"])
                        max_abs = max(abs(weight_df["weight"].min()), abs(weight_df["weight"].max()), 1e-6)
                        color_scale = alt.Scale(domain=[-max_abs, 0, max_abs], range=["#2ca02c", "#dddddd", "#d62728"])
                        chart = (
                            alt.Chart(weight_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("weight:Q", title="Weight"),
                                y=alt.Y("feature:N", sort="-x", title="Feature"),
                                color=alt.Color("weight:Q", scale=color_scale, legend=None),
                                tooltip=["feature", "weight"],
                            )
                            .properties(height=400)
                        )
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("No weights yet. The consumer will write weights.json after first message.")

                st.subheader("Historical Performance")
                trend_df = df.tail(500)
                if not trend_df.empty:
                    acc_chart = (
                        alt.Chart(trend_df)
                        .mark_line()
                        .encode(
                            x=alt.X("timestamp:T", title="Time"),
                            y=alt.Y("current_accuracy:Q", title="Accuracy"),
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(acc_chart, use_container_width=True)

                st.subheader("Recent Logs (last 10)")
                recent = df.tail(10).copy()
                st.dataframe(styled_recent(recent), use_container_width=True)

        time.sleep(refresh_rate)


render_dashboard()
