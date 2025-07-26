# syntax=docker/dockerfile:1
FROM python:3.9-slim

WORKDIR /app

# Minimal OS deps (only what you really need)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + model (model is already inside app/, so this is enough)
COPY app ./app

# Create IO dirs (they’ll be bind-mounted anyway)
RUN mkdir -p /app/input /app/output

# Optional: nicer logs
ENV PYTHONUNBUFFERED=1

# Run the CLI
ENTRYPOINT ["python", "-m", "app.main"]
