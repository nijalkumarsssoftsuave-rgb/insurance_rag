FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/data/models

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -e .

COPY app ./app
COPY scripts ./scripts

CMD ["celery", "-A", "app.workers.celery_app", "worker", "--loglevel=INFO", "--concurrency=2"]
