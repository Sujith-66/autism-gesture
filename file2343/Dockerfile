# ─────────────────────────────────────────────────────────────────────────────
# NeuroScan · ASD Behavior Detector
# Base: Python 3.11 slim — keeps image small while supporting PyTorch CPU
#
# Build:   docker build -t neuroscan-asd .
# Run:     docker run -p 5000:5000 neuroscan-asd
# GPU run: docker run --gpus all -p 5000:5000 neuroscan-asd
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System labels ─────────────────────────────────────────────────────────────
LABEL maintainer="NeuroScan"
LABEL description="ASD self-stimulatory behavior detector — MobileNetV2 + LSTM"
LABEL version="3.0"

# ── Environment flags ─────────────────────────────────────────────────────────
# Prevents .pyc clutter and ensures logs appear immediately in docker logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tell OpenCV to run headless (no display required)
    OPENCV_IO_ENABLE_OPENEXR=0 \
    # Flask
    FLASK_APP=app.py \
    FLASK_ENV=production \
    PORT=5000

# ── OS-level dependencies needed by OpenCV ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
# Copy everything except what's in .dockerignore
COPY . .

# ── Runtime directory for temporary video uploads ─────────────────────────────
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd -m -u 1000 neuroscan && \
    chown -R neuroscan:neuroscan /app
USER neuroscan

# ── Expose Flask port ─────────────────────────────────────────────────────────
EXPOSE 5000

# ── Healthcheck — Docker will mark the container unhealthy if Flask is down ───
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" \
    || exit 1

# ── Start command ─────────────────────────────────────────────────────────────
# gunicorn for production; falls back to flask dev server if gunicorn not installed
CMD ["python", "app.py"]
