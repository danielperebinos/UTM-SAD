import json
import os
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
import pika
import psycopg2
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection
import streamlit as st
from PIL import Image

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
FEEDBACK_QUEUE = os.environ.get("FEEDBACK_QUEUE", "feedback_queue")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@db:5432/postgres")


@st.cache_resource(show_spinner=False)
def feedback_channel() -> Tuple[BlockingConnection, BlockingChannel]:
    params = pika.URLParameters(RABBITMQ_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue=FEEDBACK_QUEUE, durable=True)
    return conn, ch


def send_feedback(sample_id: str, label: int) -> None:
    _, ch = feedback_channel()
    payload = {"sample_id": sample_id, "label": label}
    ch.basic_publish(
        exchange="",
        routing_key=FEEDBACK_QUEUE,
        body=json.dumps(payload),
        properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),
    )


@st.cache_data(show_spinner=False)
def load_events() -> pd.DataFrame:
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql_query(
        "SELECT sample_id, stream_id, timestamp, probability, label, image_path FROM events ORDER BY timestamp DESC",
        conn,
    )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["sample_id"])
    return df


@st.cache_data(show_spinner=False)
def load_losses() -> pd.DataFrame:
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql_query(
        "SELECT timestamp, loss, accuracy, source FROM loss_log ORDER BY timestamp ASC",
        conn,
    )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if "accuracy" not in df.columns:
        df["accuracy"] = None
    if "source" not in df.columns:
        df["source"] = "unknown"
    return df[["timestamp", "loss", "accuracy", "source"]]


@st.cache_data(show_spinner=False)
def load_image_cached(path: str, mtime: float) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def load_image(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return load_image_cached(str(path), mtime)


def analytics_tab(events: pd.DataFrame, losses: pd.DataFrame) -> None:
    st.subheader("📊 Analytics & History")

    col1, col2 = st.columns(2)
    with col1:
        if losses.empty:
            st.info("No training metrics yet.")
        else:
            chart_df = losses.set_index("timestamp")[["loss", "accuracy"]]
            st.line_chart(chart_df)
    with col2:
        if events.empty:
            st.info("No detections yet.")
        else:
            per_minute = (
                events.assign(minute=events["timestamp"].dt.floor("T"))
                .groupby("minute")["sample_id"]
                .count()
                .rename("detections_per_minute")
            )
            st.bar_chart(per_minute)

    st.markdown("---")
    st.caption("Confidence trend")
    if events.empty:
        st.info("Waiting for detections...")
    else:
        conf = events[["timestamp", "probability"]].set_index("timestamp")
        st.line_chart(conf)


def latest_per_stream(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    return events.sort_values("timestamp").drop_duplicates(subset=["stream_id"], keep="last")


def live_tab(events: pd.DataFrame, live_enabled: bool) -> None:
    st.subheader("📺 Live Streams")
    if not live_enabled:
        st.warning("Live view paused from sidebar.")
        return
    if events.empty:
        st.info("No streams yet.")
        return

    latest = latest_per_stream(events)
    num_cols = min(4, max(1, len(latest)))
    cols = st.columns(num_cols)
    for idx, (_, row) in enumerate(latest.iterrows()):
        col = cols[idx % num_cols]
        with col:
            image = load_image(Path(row.get("image_path", "")))
            st.caption(f"{row.get('stream_id', 'stream')} – {row['timestamp'].strftime('%H:%M:%S')}")
            if image is not None:
                placeholder = st.empty()
                placeholder.image(image, use_column_width=True)
            st.metric("User confidence", f"{float(row.get('probability', 0))*100:.1f}%")


def annotation_tab(events: pd.DataFrame) -> None:
    st.subheader("🏷️ Annotation & Feedback")
    if events.empty:
        st.info("Waiting for crops to annotate...")
        return
    max_items = 30
    subset = events.head(max_items)
    cols_per_row = 5
    for start in range(0, len(subset), cols_per_row):
        row_chunk = subset.iloc[start : start + cols_per_row]
        cols = st.columns(len(row_chunk))
        for col, (_, row) in zip(cols, row_chunk.iterrows()):
            image = load_image(Path(row.get("image_path", "")))
            with col:
                st.caption(f"{row.get('stream_id', '')} – {row['timestamp'].strftime('%H:%M:%S')}")
                if image is not None:
                    st.image(image, width=150)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Me", key=f"yes_{row.sample_id}"):
                        send_feedback(row.sample_id, 1)
                        st.success("Sent")
                with c2:
                    if st.button("❌ Not Me", key=f"no_{row.sample_id}"):
                        send_feedback(row.sample_id, 0)
                        st.warning("Sent")


def verification_tab(events: pd.DataFrame) -> None:
    st.subheader("🔍 History & Verification")
    if events.empty:
        st.info("No history yet.")
        return
    history = events.head(50)
    cols_per_row = 5
    for start in range(0, len(history), cols_per_row):
        chunk = history.iloc[start : start + cols_per_row]
        cols = st.columns(len(chunk))
        for col, (_, row) in zip(cols, chunk.iterrows()):
            prob = float(row.get("probability", 0))
            verdict = "Target (Me)" if prob >= 0.5 else "Unknown"
            color = "#2e8b57" if prob >= 0.5 else "#b22222"
            image = load_image(Path(row.get("image_path", "")))
            with col:
                st.markdown(
                    f"<div style='border:2px solid {color}; border-radius:6px; padding:6px;'>",
                    unsafe_allow_html=True,
                )
                if image is not None:
                    st.image(image, width=160)
                st.markdown(
                    f"<div style='color:{color}; font-weight:700;'>{verdict}</div>"
                    f"<div style='font-size:12px;'>Conf: {prob*100:.1f}%</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"{row.get('stream_id', '')} – {row['timestamp'].strftime('%H:%M:%S')}")
                st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Multi-Stream Vision Dashboard", layout="wide")
    st.title("Multi-Stream Vision Dashboard")
    st.caption("Transfer learning + active learning")

    if "live_enabled" not in st.session_state:
        st.session_state.live_enabled = True

    with st.sidebar:
        st.header("Controls")
        toggle_label = "Stop Stream" if st.session_state.live_enabled else "Start Stream"
        if st.button(toggle_label):
            st.session_state.live_enabled = not st.session_state.live_enabled
        if st.button("Clear History"):
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("TRUNCATE TABLE events;")
            cur.execute("TRUNCATE TABLE loss_log;")
            conn.commit()
            cur.close()
            conn.close()
            st.cache_data.clear()
            st.success("History cleared")

    events = load_events()
    losses = load_losses()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Analytics & History", "📺 Live Streams", "🏷️ Annotation", "🔍 Verification"]
    )
    with tab1:
        analytics_tab(events, losses)
    with tab2:
        live_tab(events, st.session_state.live_enabled)
    with tab3:
        annotation_tab(events)
    with tab4:
        verification_tab(events)

    if st.session_state.live_enabled:
        time.sleep(0.3)
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()  # type: ignore[attr-defined]
            except Exception:
                pass


if __name__ == "__main__":
    main()
