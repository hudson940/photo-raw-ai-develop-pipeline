# PhotoRAW pipeline + web UI.
# One image, two entrypoints (picked by docker-compose `command:`):
#   python -m pipeline.webui --host 0.0.0.0     review UI + share links + render jobs
#   python -m pipeline.run                      inbox watcher + AI analyze/develop worker

# --- stage 1: build the React SPA (Vite -> web/dist) ---
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install
COPY web/ ./
RUN npm run build

# --- stage 2: the Python app ---
FROM python:3.12-slim

# exiftool: fast embedded-preview extraction (Stage 2). libgl/libglib: OpenCV runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        exiftool libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline ./pipeline
# built SPA from stage 1 (served by pipeline.webui)
COPY --from=web /web/dist ./web/dist

# All state lives under /data (mount a volume). The rembg model cache goes to
# /models so it survives image rebuilds and is shared by webui + worker.
ENV PIPELINE_ROOT=/data \
    U2NET_HOME=/models/rembg \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data /models/rembg

EXPOSE 8765
CMD ["python", "-m", "pipeline.webui", "--host", "0.0.0.0"]
