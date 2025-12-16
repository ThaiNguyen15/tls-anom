# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
COPY config/ /app/config/

RUN pip install --upgrade pip && \
    pip install -e .

# Non-root user (security best practice)
RUN useradd -ms /bin/bash appuser
USER appuser

ENTRYPOINT ["tls-anom"]
CMD ["run", "--dataset", "data/raw/normal.csv", "--name", "normal", "--config", "config/default.yaml"]
