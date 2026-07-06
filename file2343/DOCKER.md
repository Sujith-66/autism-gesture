# NeuroScan · Docker Deployment Guide

## What's included

| File | Purpose |
|------|---------|
| `Dockerfile` | Single-stage Python 3.11-slim image |
| `docker-compose.yml` | One-command CPU & GPU service definitions |
| `requirements.txt` | Pinned deps — CPU-only torch for a lean image |
| `.dockerignore` | Excludes training data, notebooks, and caches |

---

## Prerequisites

- [Docker Desktop](https://docs.docker.com/get-docker/) (Windows/Mac) or Docker Engine (Linux)
- For GPU mode: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

## Quick start (CPU)

```bash
# 1. Enter the project folder
cd file2343

# 2. Build the image  (~3–5 min first time; cached after that)
docker compose build

# 3. Start the container
docker compose up

# 4. Open your browser
#    http://localhost:5000
```

To run detached (in background):
```bash
docker compose up -d
```

To stop:
```bash
docker compose down
```

---

## GPU mode

Requires NVIDIA GPU + NVIDIA Container Toolkit installed.

```bash
docker compose --profile gpu up
```

The `neuroscan-gpu` service maps `--gpus all` into the container.
PyTorch's `BehaviorDetector` auto-detects CUDA and uses it.

> **Note:** `requirements.txt` ships CPU-only PyTorch to keep image size small (~1.8 GB).
> For GPU builds, edit `requirements.txt`:
> - Remove `+cpu` suffix from `torch` and `torchvision`
> - Remove the `--extra-index-url` line
> Then rebuild: `docker compose build`

---

## Build the image manually (without Compose)

```bash
docker build -t neuroscan-asd .
docker run -p 5000:5000 neuroscan-asd
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Port Flask listens on |
| `FLASK_ENV` | `production` | Set to `development` for debug mode |

Override at runtime:
```bash
docker run -p 8080:8080 -e PORT=8080 neuroscan-asd
```

---

## Healthcheck

Docker automatically polls `/health` every 30 seconds.
```bash
# Check container health status
docker ps
# HEALTHY means Flask + model loaded successfully
```

Manual check:
```bash
curl http://localhost:5000/health
# {"status": "ok", "classes": ["Arm Flapping", "Head Banging", "Spinning"]}
```

---

## Uploading / persistent data

Temporary video uploads are stored at `/app/uploads` inside the container.
Docker Compose mounts a named volume `uploads_data` so files survive restarts.

To wipe all uploaded videos:
```bash
docker compose down -v   # removes the volume too
```

---

## Checking logs

```bash
docker compose logs -f neuroscan
```

---

## Image size tips

| Build variant | Approx size |
|--------------|-------------|
| CPU (default) | ~1.8 GB |
| GPU (CUDA 12.1) | ~4.5 GB |

The `.dockerignore` excludes `data/` (training videos), `*.npy` files, and notebooks,
saving ~400 MB from the build context.

---

## Troubleshooting

**Port already in use**
```bash
# Change host port:
docker run -p 5001:5000 neuroscan-asd
```

**OpenCV error: `libGL.so.1` not found**
Already handled — the Dockerfile installs `libgl1-mesa-glx`. If you build a custom base image, add this package.

**Model not found warning**
`lstm_model.pth` is included in the image. If you see a warning, ensure the file exists in the project directory before building.

**Container keeps restarting**
```bash
docker logs neuroscan-asd
# Usually a missing dependency or model file
```
