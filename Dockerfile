# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy metadata first (better cache)
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
COPY config/ /app/config/

RUN pip install --upgrade pip && \
    pip install -e .

# Create non-root user
RUN useradd -m -u 10001 appuser

# Create writable directories for PVC mounts
RUN mkdir -p /data /outputs && \
    chown -R appuser:appuser /app /data /outputs

USER appuser

# Document volumes (PVC in K8s)
VOLUME ["/data", "/outputs"]

ENTRYPOINT ["tls-anom"]