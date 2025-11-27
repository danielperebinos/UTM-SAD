import base64
import io
import json
import logging
import os
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pika
import psycopg2
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection
from torch.utils.data import DataLoader
from torchvision import models, transforms
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s - %(message)s",
)

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
VIDEO_QUEUE = os.environ.get("VIDEO_QUEUE", "video_stream")
FEEDBACK_QUEUE = os.environ.get("FEEDBACK_QUEUE", "feedback_queue")
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
REFERENCE_DIR = os.path.join(DATA_DIR, "reference")
CROPS_DIR = os.path.join(DATA_DIR, "crops")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@db:5432/postgres")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = os.environ.get("MODEL_NAME", "resnet18").lower()
REF_EPOCHS = int(os.environ.get("REF_EPOCHS", "2"))
REF_MAX_PER_CLASS = int(os.environ.get("REF_MAX_PER_CLASS", "200"))
REF_FRAME_STRIDE = int(os.environ.get("REF_FRAME_STRIDE", "10"))
USE_YOLO_FOR_REF = os.environ.get("USE_YOLO_FOR_REF", "1") != "0"
REF_YOLO_MODEL = os.environ.get("REF_YOLO_MODEL", "yolov8n.pt")

os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REFERENCE_DIR, exist_ok=True)

recent_samples: "OrderedDict[str, tuple[torch.Tensor, str]]" = OrderedDict()
recent_lock = threading.Lock()
log_lock = threading.Lock()
loss_log_lock = threading.Lock()
MAX_CACHE = 256


def db_connect():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn


def ensure_tables() -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            sample_id TEXT PRIMARY KEY,
            stream_id TEXT,
            timestamp TIMESTAMPTZ,
            probability DOUBLE PRECISION,
            label TEXT,
            image_path TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS loss_log (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ,
            loss DOUBLE PRECISION,
            accuracy DOUBLE PRECISION,
            source TEXT
        );
        """
    )
    cur.close()
    conn.close()


def append_event(
    sample_id: str,
    stream_id: str,
    timestamp: str,
    prob: float,
    label: str,
    image_path: str,
) -> None:
    with log_lock:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO events (sample_id, stream_id, timestamp, probability, label, image_path)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (sample_id) DO UPDATE
            SET stream_id = EXCLUDED.stream_id,
                timestamp = EXCLUDED.timestamp,
                probability = EXCLUDED.probability,
                label = EXCLUDED.label,
                image_path = EXCLUDED.image_path;
            """,
            (sample_id, stream_id, timestamp, prob, label, image_path),
        )
        cur.close()
        conn.close()


def append_loss(timestamp: str, loss: float, accuracy: float, source: str) -> None:
    with loss_log_lock:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO loss_log (timestamp, loss, accuracy, source)
            VALUES (%s, %s, %s, %s);
            """,
            (timestamp, loss, accuracy, source),
        )
        cur.close()
        conn.close()


def build_model(model_name: str) -> tuple[nn.Module, transforms.Compose, List[nn.Parameter]]:
    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        transform = weights.transforms()
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
    else:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        transform = weights.transforms()
        for name, param in model.named_parameters():
            if name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)

    trainable = [p for p in model.parameters() if p.requires_grad]
    return model, transform, trainable


model, transform, trainable_params = build_model(MODEL_NAME)
model = model.to(DEVICE)
optimizer = optim.Adam(trainable_params, lr=1e-4)
criterion = nn.CrossEntropyLoss()

ref_yolo_model = None
if USE_YOLO_FOR_REF:
    try:
        ref_yolo_model = YOLO(REF_YOLO_MODEL)
        logging.info("Loaded YOLO model %s for reference bootstrapping", REF_YOLO_MODEL)
    except Exception as exc:
        logging.warning("Could not load YOLO for reference data: %s", exc)


def decode_image(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64, validate=True)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return image


def preprocess(image: Image.Image) -> torch.Tensor:
    return transform(image)


def cache_sample(sample_id: str, tensor: torch.Tensor, stream_id: str) -> None:
    with recent_lock:
        if sample_id in recent_samples:
            recent_samples.move_to_end(sample_id)
        recent_samples[sample_id] = (tensor.cpu(), stream_id)
        if len(recent_samples) > MAX_CACHE:
            recent_samples.popitem(last=False)


def fetch_sample(sample_id: str) -> Optional[tuple[torch.Tensor, Optional[str]]]:
    with recent_lock:
        tensor = recent_samples.get(sample_id)
        if tensor is not None:
            return tensor
    path = os.path.join(CROPS_DIR, f"{sample_id}.jpg")
    if os.path.exists(path):
        img = Image.open(path).convert("RGB")
        return preprocess(img), None
    return None


def safe_crop(frame: np.ndarray, bbox: Iterable[float]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def extract_crops_from_frame(frame: np.ndarray) -> List[Image.Image]:
    crops: List[Image.Image] = []
    if ref_yolo_model is not None:
        try:
            results = ref_yolo_model(frame, classes=[0], verbose=False)[0]
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    crop = safe_crop(frame, (x1, y1, x2, y2))
                    if crop is not None and crop.size > 0:
                        crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        except Exception as exc:
            logging.warning("YOLO crop failed, fallback to full frame: %s", exc)

    if not crops:
        crops.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    return crops


def load_reference_dir(dir_path: Path, label: int) -> List[Tuple[torch.Tensor, int]]:
    samples: List[Tuple[torch.Tensor, int]] = []
    if not dir_path.exists():
        logging.info("Reference path missing: %s", dir_path)
        return samples

    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for file in sorted(dir_path.glob("*")):
        if len(samples) >= REF_MAX_PER_CLASS:
            break
        suffix = file.suffix.lower()
        if suffix in video_exts:
            cap = cv2.VideoCapture(str(file))
            frame_idx = 0
            while cap.isOpened() and len(samples) < REF_MAX_PER_CLASS:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % REF_FRAME_STRIDE != 0:
                    frame_idx += 1
                    continue
                frame_idx += 1
                for crop in extract_crops_from_frame(frame):
                    samples.append((preprocess(crop), label))
                    if len(samples) >= REF_MAX_PER_CLASS:
                        break
            cap.release()
        elif suffix in image_exts:
            frame = cv2.imread(str(file))
            if frame is None:
                continue
            for crop in extract_crops_from_frame(frame):
                samples.append((preprocess(crop), label))
        else:
            continue
    logging.info("Loaded %d reference samples from %s", len(samples), dir_path)
    return samples


def train_reference() -> None:
    me_dir = Path(REFERENCE_DIR) / "me"
    not_me_dir = Path(REFERENCE_DIR) / "not_me"

    samples = load_reference_dir(me_dir, 1) + load_reference_dir(not_me_dir, 0)
    if not samples:
        logging.info("No reference data found, skipping bootstrap training.")
        return

    loader = DataLoader(samples, batch_size=min(16, len(samples)), shuffle=True)
    model.train()
    for epoch in range(REF_EPOCHS):
        for batch_tensors, batch_labels in loader:
            optimizer.zero_grad()
            logits = model(batch_tensors.to(DEVICE))
            loss = criterion(logits, batch_labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds.cpu() == batch_labels).float().mean().item()
            append_loss(datetime.now(timezone.utc).isoformat(), loss.item(), acc, "bootstrap")
        logging.info("Bootstrap epoch %d/%d complete", epoch + 1, REF_EPOCHS)


def train_on_feedback(sample_id: str, label: int) -> None:
    sample = fetch_sample(sample_id)
    if sample is None:
        logging.warning("Feedback sample %s missing image/tensor", sample_id)
        return
    tensor, stream_id = sample

    model.train()
    optimizer.zero_grad()
    batch = tensor.unsqueeze(0).to(DEVICE)
    target = torch.tensor([label], dtype=torch.long, device=DEVICE)
    logits = model(batch)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        acc = float((preds == target).float().item())
        prob = torch.softmax(logits, dim=1)[0, 1].item()
    ts = datetime.now(timezone.utc).isoformat()
    append_loss(ts, loss.item(), acc, "feedback")
    append_event(
        sample_id,
        stream_id or "feedback",
        ts,
        prob,
        str(int(label)),
        f"{CROPS_DIR}/{sample_id}.jpg",
    )
    logging.info("Trained on feedback for %s label=%s loss=%.4f acc=%.2f", sample_id, label, loss.item(), acc)


def connect_rabbitmq() -> tuple[BlockingConnection, BlockingChannel]:
    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    return connection, channel


def handle_video_message(ch: BlockingChannel, method, properties, body: bytes) -> None:
    try:
        message = json.loads(body)
        image_b64 = message["image_base64"]
        stream_id = message.get("stream_id", "unknown")
        timestamp = message.get("timestamp", datetime.now(timezone.utc).isoformat())
        sample_id = message.get("sample_id") or f"{stream_id}_{uuid.uuid4().hex[:8]}"
    except Exception as exc:
        logging.error("Invalid message: %s", exc)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    try:
        image = decode_image(image_b64)
    except Exception as exc:
        logging.error("Base64 decode failed for %s: %s", sample_id, exc)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    tensor = preprocess(image)
    cache_sample(sample_id, tensor, stream_id)

    model.eval()
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(DEVICE))
        prob = torch.softmax(logits, dim=1)[0, 1].item()

    image_path = os.path.join(CROPS_DIR, f"{sample_id}.jpg")
    try:
        image.save(image_path, format="JPEG")
    except Exception as exc:
        logging.warning("Failed to save crop for %s: %s", sample_id, exc)

    append_event(sample_id, stream_id, timestamp, prob, "unlabeled", image_path)
    logging.info("Processed sample %s prob=%.3f", sample_id, prob)
    ch.basic_ack(delivery_tag=method.delivery_tag)


def feedback_consumer() -> None:
    connection, channel = connect_rabbitmq()
    channel.queue_declare(queue=FEEDBACK_QUEUE, durable=True)

    def _callback(ch: BlockingChannel, method, properties, body: bytes) -> None:
        try:
            msg = json.loads(body)
            sample_id = msg["sample_id"]
            label = int(msg["label"])
        except Exception as exc:
            logging.error("Invalid feedback message: %s", exc)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        train_on_feedback(sample_id, label)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=FEEDBACK_QUEUE, on_message_callback=_callback, auto_ack=False)
    logging.info("Feedback consumer started")
    try:
        channel.start_consuming()
    finally:
        channel.close()
        connection.close()


def main() -> None:
    ensure_tables()

    logging.info("Starting bootstrap training from %s", REFERENCE_DIR)
    train_reference()

    feedback_thread = threading.Thread(target=feedback_consumer, name="feedback-thread", daemon=True)
    feedback_thread.start()

    connection, channel = connect_rabbitmq()
    channel.queue_declare(queue=VIDEO_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=VIDEO_QUEUE, on_message_callback=handle_video_message, auto_ack=False)
    logging.info("Vision processor consuming from %s on %s", VIDEO_QUEUE, DEVICE)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logging.info("Stopping consumer")
    finally:
        try:
            channel.close()
            connection.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
