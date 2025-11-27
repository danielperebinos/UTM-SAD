# UTM-SAD Projects

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-FF6600.svg)](https://www.rabbitmq.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791.svg)](https://www.postgresql.org/)

Collection of two distributed systems built for the SAD coursework:
- **real-time-churn-app**: Streaming churn detection for banking data using River (online learning) with RabbitMQ and Streamlit.
- **multi-stream-vision-app**: Multi-camera people detection and identification with YOLOv8, transfer learning, active feedback, RabbitMQ, Postgres logging, and Streamlit UI.

## Repository Layout
```
notebooks/                 # Research notebooks
real-time-churn-app/       # Streaming churn detection stack
multi-stream-vision-app/   # Multi-stream computer vision stack
```

## Quick Start
Prerequisites: Docker + Docker Compose.

- real-time-churn-app:
  ```bash
  cd real-time-churn-app
  docker compose up --build
  ```
  Dashboard: http://localhost:8501 | RabbitMQ: http://localhost:15672 (guest/guest)

- multi-stream-vision-app:
  ```bash
  cd multi-stream-vision-app
  docker compose up --build
  ```
  Dashboard: http://localhost:8501 | RabbitMQ: http://localhost:15672 | Postgres: localhost:5432 (postgres/postgres)

## Documentation
- Real-time churn stack: `real-time-churn-app/README.md`
- Multi-stream vision stack: `multi-stream-vision-app/README.md`