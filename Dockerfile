# Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY inference_service /app/inference_service
COPY neural_network /app/neural_network

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn inference_service.server:app --host 0.0.0.0 --port ${PORT:-8080}"]