# Real-Time Banking Churn Detection System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-FF6600.svg)](https://www.rabbitmq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](#)

Real-time churn detection for banking customers using online (incremental) learning with River. Events stream through RabbitMQ; the consumer learns and explains each decision on the fly, and the Streamlit dashboard visualizes performance and XAI signals in real time.

## Architecture
```mermaid
flowchart LR
    A[CSV Data] --> B[Producer]
    B --> C[RabbitMQ]
    C --> D[Consumer (River Model)]
    D --> E[Shared Logs (CSV/JSON)]
    E --> F[Streamlit Dashboard]
```

## Key Features
- Online learning with River (instance-by-instance updates; no full retrains)
- Real-time XAI (live feature weights and per-event insight)
- Drift monitoring via continuous accuracy and probability tracking
- Fully containerized (Docker Compose for broker, producer, consumer, dashboard)

## Tech Stack
- Python 3.11, River, pandas, numpy
- RabbitMQ + pika
- Streamlit + Altair
- Docker & Docker Compose
- uv for Python dependency management

## Installation & Usage
Prerequisites: Docker and Docker Compose.

1) Start the stack:
```bash
cd real-time-churn-app
docker compose up --build
```

2) Access points:
- Dashboard: http://localhost:8501
- RabbitMQ UI: http://localhost:15672 (guest / guest by default)

Data source: `data/Churn_Modelling.csv` (mounted read-only into the producer).

## Project Structure
```
real-time-churn-app/
+-- consumer/
¦   +-- Dockerfile
¦   +-- main.py
+-- producer/
¦   +-- Dockerfile
¦   +-- main.py
+-- dashboard/
¦   +-- Dockerfile
¦   +-- main.py
+-- data/
¦   +-- Churn_Modelling.csv
+-- docker-compose.yaml
+-- requirements.txt
+-- README.md
```

## Dashboard Preview
![Dashboard Screenshot](assets/dashboard.png)