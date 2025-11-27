import base64
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import List, Tuple
import uuid

import cv2
import numpy as np
import pika
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s - %(message)s",
)

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "video_stream")
STREAM_SOURCES_ENV = os.environ.get("STREAM_SOURCES", "data/sample.mp4")
FPS_LIMIT = float(os.environ.get("FPS_LIMIT", "5"))
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")

model_lock = threading.Lock()
model = YOLO(YOLO_MODEL)


def parse_sources(raw: str) -> List[Tuple[str, str]]:
    """
    Parse a comma-separated list of sources.
    Supported formats:
    - "cam1|path_or_rtsp" (preferred, avoids conflict with URLs)
    - "cam1=path_or_rtsp"
    - "path_or_rtsp" (stream id auto-assigned)
    """
    sources: List[Tuple[str, str]] = []
    for idx, entry in enumerate(raw.split(",")):
        entry = entry.strip()
        if not entry:
            continue
        if "|" in entry:
            stream_id, path = entry.split("|", 1)
        elif "=" in entry:
            stream_id, path = entry.split("=", 1)
        else:
            stream_id, path = f"cam_{idx + 1}", entry
        sources.append((stream_id.strip(), path.strip()))
    return sources


def connect_rabbitmq() -> tuple[BlockingConnection, BlockingChannel]:
    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    return connection, channel


def crop_person(frame: np.ndarray, bbox: List[int]) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def encode_image(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    return base64.b64encode(buffer).decode("utf-8")


def process_stream(stream_id: str, source: str, frame_interval: float) -> None:
    logging.info("Starting stream %s from %s", stream_id, source)
    connection, channel = connect_rabbitmq()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("Unable to open source %s", source)
        return

    last_sent = 0.0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                logging.warning("Stream %s ended or failed to read frame", stream_id)
                break

            now = time.time()
            if now - last_sent < frame_interval:
                continue

            last_sent = now
            with model_lock:
                results = model(frame, classes=[0], verbose=False)[0]

            if results.boxes is None or len(results.boxes) == 0:
                continue

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                sample_id = f"{stream_id}_{int(now * 1000)}_{uuid.uuid4().hex[:8]}"
                crop = crop_person(frame, [x1, y1, x2, y2])
                if crop is None or crop.size == 0:
                    continue
                try:
                    image_b64 = encode_image(crop)
                except ValueError as err:
                    logging.warning("Encoding failed for %s: %s", stream_id, err)
                    continue

                payload = {
                    "sample_id": sample_id,
                    "stream_id": stream_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "image_base64": image_b64,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
                channel.basic_publish(
                    exchange="",
                    routing_key=QUEUE_NAME,
                    body=json.dumps(payload),
                    properties=pika.BasicProperties(
                        delivery_mode=2,
                        content_type="application/json",
                        app_id="video_ingest",
                    ),
                )
    finally:
        cap.release()
        try:
            channel.close()
        except Exception:
            pass
        try:
            connection.close()
        except Exception:
            pass
        logging.info("Stream %s stopped", stream_id)


def main() -> None:
    sources = parse_sources(STREAM_SOURCES_ENV)
    if not sources:
        raise RuntimeError("No sources provided. Set STREAM_SOURCES env var.")

    frame_interval = 1.0 / FPS_LIMIT if FPS_LIMIT > 0 else 0.0
    threads: list[threading.Thread] = []
    for stream_id, source in sources:
        t = threading.Thread(
            target=process_stream, args=(stream_id, source, frame_interval), daemon=True
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
