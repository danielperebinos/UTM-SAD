import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pika
from pika.exceptions import AMQPConnectionError, ChannelWrongStateError, StreamLostError
from river import compose, linear_model, metrics, preprocessing


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
QUEUE_NAME = os.getenv("QUEUE_NAME", "churn_stream")
LOG_DIR = Path(os.getenv("LOG_DIR", "/app/logs"))
LOG_CSV = LOG_DIR / "data_log.csv"
WEIGHTS_JSON = LOG_DIR / "weights.json"

CATEGORICAL_FEATURES = ["Geography", "Gender"]
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
TARGET_COL = "Exited"


def build_model() -> Tuple[compose.Pipeline, linear_model.LogisticRegression, metrics.Accuracy]:
    # Compose transformers so categorical features are one-hot encoded and numeric features are scaled.
    numeric = compose.Select(*NUMERIC_FEATURES) | preprocessing.StandardScaler()
    categorical = compose.Select(*CATEGORICAL_FEATURES) | preprocessing.OneHotEncoder()
    log_reg = linear_model.LogisticRegression()
    model = compose.TransformerUnion(numeric, categorical) | log_reg
    return model, log_reg, metrics.Accuracy()


def connect_rabbitmq() -> Tuple[pika.BlockingConnection, pika.adapters.blocking_connection.BlockingChannel]:
    while True:
        try:
            params = pika.ConnectionParameters(host=RABBITMQ_HOST, heartbeat=600, blocked_connection_timeout=300)
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.basic_qos(prefetch_count=1)
            print(f"[consumer] Connected to RabbitMQ at '{RABBITMQ_HOST}', queue '{QUEUE_NAME}' ready.")
            return connection, channel
        except AMQPConnectionError as exc:
            print(f"[consumer] RabbitMQ unavailable ({exc}); retrying in 3s...")
            time.sleep(3)


def ensure_log_header() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_CSV.exists():
        with LOG_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "row_id",
                    "true_label",
                    "prediction",
                    "probability",
                    "current_accuracy",
                    "top_feature_name",
                    "top_feature_weight",
                ],
            )
            writer.writeheader()


def explain_weights(weights: Dict[str, float], limit: int = 5) -> str:
    if not weights:
        return "weights not learned yet"
    top_weights = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)[:limit]
    return "; ".join(f"{name}={value:.3f}" for name, value in top_weights)


def write_weights_snapshot(weights: Dict[str, float]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = WEIGHTS_JSON.with_suffix(".tmp")
    with tmp_path.open("w") as f:
        json.dump(weights, f)
    tmp_path.replace(WEIGHTS_JSON)


def append_log(row: Dict[str, str]) -> None:
    ensure_log_header()
    with LOG_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "row_id",
                "true_label",
                "prediction",
                "probability",
                "current_accuracy",
                "top_feature_name",
                "top_feature_weight",
            ],
        )
        writer.writerow(row)


def main() -> None:
    model, log_reg, metric = build_model()
    print("[consumer] Model initialized (OneHotEncoder + StandardScaler -> LogisticRegression).")

    connection, channel = connect_rabbitmq()
    ensure_log_header()
    row_id = 0

    def handle_message(ch, method, properties, body) -> None:
        nonlocal row_id
        row_id += 1
        try:
            payload = json.loads(body)
            features = payload.get("features", {})
            label = int(payload.get("label", 0))
        except Exception as exc:
            print(f"[consumer] Failed to parse message ({exc}); discarding.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        # Predict -> Explain -> Learn cycle for real-time feedback.
        proba_true = model.predict_proba_one(features).get(True, 0.0)
        prediction = model.predict_one(features)
        weight_snapshot = dict(log_reg.weights)
        weights_view = explain_weights(weight_snapshot)

        model.learn_one(features, label)
        metric.update(label, prediction)
        current_acc = metric.get()

        # Persist metrics and weights for dashboard consumption.
        append_log(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "row_id": row_id,
                "true_label": label,
                "prediction": prediction if prediction is not None else "",
                "probability": f"{proba_true:.4f}",
                "current_accuracy": f"{current_acc:.4f}" if current_acc is not None else "",
                "top_feature_name": weight_snapshot and max(weight_snapshot, key=lambda k: abs(weight_snapshot[k]))
                or "",
                "top_feature_weight": weight_snapshot and f"{weight_snapshot[max(weight_snapshot, key=lambda k: abs(weight_snapshot[k]))]:.4f}"
                or "",
            }
        )
        write_weights_snapshot(weight_snapshot)

        print(
            "[consumer] row_id={row_id} pred={pred} prob_true={prob:.3f} label={label} "
            "acc={acc} | top_weights: {weights}".format(
                row_id=row_id,
                pred=prediction,
                prob=proba_true,
                label=label,
                acc=f"{current_acc:.3f}" if current_acc is not None else "n/a",
                weights=weights_view,
            )
        )

        ch.basic_ack(delivery_tag=method.delivery_tag)

    while True:
        try:
            channel.basic_consume(queue=QUEUE_NAME, on_message_callback=handle_message, auto_ack=False)
            channel.start_consuming()
        except (AMQPConnectionError, ChannelWrongStateError, StreamLostError) as exc:
            print(f"[consumer] Connection lost ({exc}); reconnecting.")
            try:
                connection.close()
            except Exception:
                pass
            connection, channel = connect_rabbitmq()
        except KeyboardInterrupt:
            print("[consumer] Stopping consumer.")
            try:
                channel.stop_consuming()
                connection.close()
            finally:
                break


if __name__ == "__main__":
    main()
