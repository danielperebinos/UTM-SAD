FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN pip install uv && uv pip install --system -r requirements.txt

COPY consumer/main.py /app/main.py

CMD ["python", "/app/main.py"]
