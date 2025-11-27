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


LOG_FIELDS = [
    "timestamp",
    "row_id",
    "customer_id",
    "geography",
    "gender",
    "age",
    "balance",
    "credit_score",
    "tenure",
    "num_products",
    "is_active_member",
    "true_label",
    "prediction",
    "probability",
    "current_accuracy",
    "top_feature_name",
    "top_feature_weight",
]


def ensure_log_header() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_header = False
    if not LOG_CSV.exists():
        write_header = True
    else:
        try:
            with LOG_CSV.open("r") as f:
                first_line = f.readline().strip().split(",")
            if set(first_line) != set(LOG_FIELDS):
                write_header = True
        except Exception:
            write_header = True
    if write_header:
        with LOG_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
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
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
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

        # Extract feature fields for richer logging.
        top_feature = ""
        top_weight = ""
        if weight_snapshot:
            top_feature = max(weight_snapshot, key=lambda k: abs(weight_snapshot[k]))
            top_weight = f"{weight_snapshot[top_feature]:.4f}"

        append_log(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "row_id": row_id,
                "customer_id": features.get("CustomerId", features.get("RowNumber", row_id)),
                "geography": features.get("Geography", ""),
                "gender": features.get("Gender", ""),
                "age": features.get("Age", ""),
                "balance": features.get("Balance", ""),
                "credit_score": features.get("CreditScore", ""),
                "tenure": features.get("Tenure", ""),
                "num_products": features.get("NumOfProducts", ""),
                "is_active_member": features.get("IsActiveMember", ""),
                "true_label": label,
                "prediction": prediction if prediction is not None else "",
                "probability": f"{proba_true:.4f}",
                "current_accuracy": f"{current_acc:.4f}" if current_acc is not None else "",
                "top_feature_name": top_feature,
                "top_feature_weight": top_weight,
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
