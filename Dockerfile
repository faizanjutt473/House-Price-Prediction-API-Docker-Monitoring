# ---------- Stage 1: install dependencies and train the model ----------
# Training inside the build guarantees the model is created with the exact
# scikit-learn version that serves it (pickle files are version sensitive).
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ app/
COPY train.py .
RUN python train.py --out model

# ---------- Stage 2: small runtime image ----------
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/model /app/model
COPY app/ app/
RUN useradd --create-home appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1
# One worker on purpose: latency and drift statistics live in memory and must be in one process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
