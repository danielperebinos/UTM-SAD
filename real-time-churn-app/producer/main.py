import json
import os
import random
import time
from typing import Any, Dict, Tuple

import pandas as pd
import pika
from pika.exceptions import AMQPConnectionError, ChannelWrongStateError, StreamLostError


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
QUEUE_NAME = os.getenv("QUEUE_NAME", "churn_stream")
DATA_PATH = os.getenv("DATA_PATH", "/app/data/Churn_Modelling.csv")
DELAY_MIN = float(os.getenv("STREAM_DELAY_MIN", "0.5"))
DELAY_MAX = float(os.getenv("STREAM_DELAY_MAX", "1.0"))
TARGET_COL = "Exited"


def load_data() -> Tuple[pd.DataFrame, list]:
    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")
    feature_cols = [col for col in df.columns if col != TARGET_COL]
    return df, feature_cols


def connect_rabbitmq() -> Tuple[pika.BlockingConnection, pika.adapters.blocking_connection.BlockingChannel]:
    while True:
        try:
            params = pika.ConnectionParameters(host=RABBITMQ_HOST, heartbeat=600, blocked_connection_timeout=300)
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            print(f"[producer] Connected to RabbitMQ at '{RABBITMQ_HOST}', queue '{QUEUE_NAME}' ready.")
            return connection, channel
        except AMQPConnectionError as exc:
            print(f"[producer] RabbitMQ unavailable ({exc}); retrying in 3s...")
            time.sleep(3)


def to_serializable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if pd.isna(value):
        return None
    return value


def main() -> None:
    df, feature_cols = load_data()
    print(f"[producer] Loaded dataset with {len(df)} rows and features: {feature_cols}")

    connection, channel = connect_rabbitmq()

    while True:
        for _, row in df.iterrows():
            message = {
                "features": {col: to_serializable(row[col]) for col in feature_cols},
                "label": int(row[TARGET_COL]),
            }

            try:
                channel.basic_publish(
                    exchange="",
                    routing_key=QUEUE_NAME,
                    body=json.dumps(message),
                    properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),
                )
                print(f"[producer] Published message for customer (label={message['label']}).")
            except (AMQPConnectionError, ChannelWrongStateError, StreamLostError) as exc:
                print(f"[producer] Publish failed ({exc}); attempting to reconnect.")
                try:
                    connection.close()
                except Exception:
                    pass
                connection, channel = connect_rabbitmq()
                continue

            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


if __name__ == "__main__":
    main()
