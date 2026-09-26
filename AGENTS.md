# AGENTS.md — PhotoRAW on the Capibara VPS

Operational guide for running and maintaining this deployment. Read this first when
working on the server.

## Where things run

| | |
|---|---|
| **Host** | `anderson@10.200.1.1` (Ubuntu) |
| **Project dir** | `/home/anderson/photos_capibara` |
| **Public URL** | https://fotos.capibaraeventos.com (DNS → this server; nginx proxy is set up separately) |
| **Web UI / API** | container `webui`, published on host port **8765** |
| **Auth** | Keycloak (`keycloak` + `kc-postgres` containers), realm `photoraw` |
| **Database** | **SQLite** at `/home/anderson/photos_capibara/data/pipeline.db` (on the `data` Docker volume via the bind path — see Storage) |
| **Object storage** | starts in **local** mode; switch to S3 when creds are set (see below) |

The stack is defined in `docker-compose.yml`. This deployment runs the **`auth`
profile only** (Keycloak). The `postgres` and `local-s3` profiles are intentionally
NOT used here (SQLite DB, external S3).

Containers that run here: `webui`, `worker`, `keycloak`, `kc-postgres`.

## Everyday commands

Run everything from the project dir:

```bash
cd /home/anderson/photos_capibara

# start / update the stack (auth profile = Keycloak)
docker compose --profile auth up -d --build

# status + health
docker compose --profile auth ps

# logs
docker compose --profile auth logs -f webui        # the UI/API
docker compose --profile auth logs -f worker       # inbox watcher + analyze/develop
docker compose --profile auth logs -f keycloak

# restart one service
docker compose --profile auth restart webui

# stop everything (keeps volumes/data)
docker compose --profile auth down
```

> Always include `--profile auth`, otherwise `docker compose down`/`ps` won't see the
> Keycloak containers.

## Configuration (`.env`)

`.env` lives in the project dir (chmod 600, never committed). Key values:

- `PIPELINE_DATABASE_URL=` — **empty on purpose** → SQLite. Set a `postgresql://…` URL
  (and add `--profile postgres`, or point at an external DB) to switch to Postgres.
- `KEYCLOAK_URL=http://keycloak:8080` — the app reaches Keycloak over the Docker network
  (in-app login form → direct grant), so Keycloak does **not** need a public port for login.
- `KEYCLOAK_PORT=8085` — host port for the Keycloak admin console (8080 was taken by
  Nextcloud on this box). Optional; only needed if you want the Keycloak console directly.
- `PIPELINE_STORAGE=local` — see Storage below.
- Secrets (`KEYCLOAK_CLIENT_SECRET`, `SESSION_SECRET`, `KC_ADMIN_PASSWORD`,
  `KC_DB_PASSWORD`) were generated at deploy time. `KEYCLOAK_CLIENT_SECRET` must match
  `keycloak/realm-photoraw.json`.

### Must fill in to be fully functional

- **`ANTHROPIC_API_KEY`** — required for the worker to *analyze new uploads* (Stage 3
  AI vision). The worker runs a `verify_vision` canary at startup and **exits if the key
  is invalid**, so it is **currently stopped** (deployed with a placeholder key).
  Browsing/downloading the 808 already-developed photos works without it. To enable
  ingestion of new photos:
  ```bash
  # edit .env: ANTHROPIC_API_KEY=sk-ant-...
  docker compose --profile auth up -d worker      # start it
  docker compose --profile auth logs -f worker     # confirm no 401
  ```
- **S3 credentials** — see Storage.

After editing `.env`: `docker compose --profile auth up -d` (recreates affected
containers).

## Storage: local → S3

Currently `PIPELINE_STORAGE=local` — all files live on the `data` volume (which is
what was rsynced from the workstation). To move durable artifacts (RAW archive,
previews, outputs, thumbnails, originals, masks) to your S3 service:

1. In `.env` set:
   ```
   PIPELINE_STORAGE=s3
   S3_ENDPOINT_URL=https://<your-endpoint>
   S3_ACCESS_KEY=...
   S3_SECRET_KEY=...
   S3_BUCKET=...
   #S3_REGION=...        # if required by the provider
   #S3_PREFIX=capibara   # optional key prefix
   ```
2. `docker compose --profile auth up -d webui worker`
3. Backfill the bucket from the local cache (one-time), e.g. with the `aws`/`mc` CLI, or
   just let files upload lazily as they are re-produced. The local `data` volume then
   becomes a rebuildable cache; the bucket is the source of truth.

> Do NOT set `PIPELINE_STORAGE=s3` with blank/invalid creds — the photo list calls S3
> per photo and will error. Keep `local` until the creds are real.

### Client-side encryption (E2E)

Optionally encrypt objects so the bucket only ever holds ciphertext — the S3 provider
(and anyone who steals the S3 keys) can't read the photos. Set one key in `.env`:

```
STORAGE_ENCRYPTION_KEY=$(openssl rand -base64 32)   # 32-byte key; or a passphrase
```

Files are encrypted (AES-256-GCM, chunked) on this host **before** upload and decrypted
after download; the local `data` cache stays plaintext (the pipeline needs real files).

- **Guard this key like a password. If you lose it, every remote-only photo is gone.** Keep
  it out of git; back it up somewhere safe. Changing it makes previously-uploaded objects
  unreadable.
- Turning it on over a bucket that already has plaintext objects is safe: old files still
  read (auto-detected and passed through), new uploads are encrypted. To encrypt the
  existing ones, re-upload them (they re-upload as they're re-produced, or force a put).
- Only the payload is encrypted — object **keys/paths** (e.g. `output/IMG_1.jpg`) are not,
  so filenames are still visible in the bucket listing.

## Database

SQLite, single file. Inspect / back up:

```bash
# read-only peek
docker compose --profile auth exec webui \
  python -c "import sqlite3;print(sqlite3.connect('/data/pipeline.db').execute(\
  'select state,count(*) from photos group by state').fetchall())"

# consistent backup (safe while running)
sqlite3 data/pipeline.db ".backup 'data/pipeline.backup.db'"
```

The schema auto-migrates on startup (`pipeline/db.py`). To migrate to Postgres later,
stand up the `postgres` profile and copy rows over (the app's SQL is portable via the
`Conn` wrapper in `pipeline/db.py`).

## Users & auth

- Roles: `super_admin` (manages users, sees all photos) and `editor` (sees only their
  own photos; inbox subfolder `inbox/<username>/` sets ownership).
- Manage users from the **Users** panel in the web UI (super admin) — it drives the
  Keycloak Admin API server-side. You generally don't need the Keycloak console.
- The initial super admin is `admin` (password set in `keycloak/realm-photoraw.json` at
  deploy). Keycloak's own master admin is `KC_ADMIN`/`KC_ADMIN_PASSWORD` (console at
  `http://<host>:8085` if `KEYCLOAK_PORT=8085`).

## nginx (to be configured)

Docker publishes the app on `0.0.0.0:8765`. Point the vhost at it. Minimal proxy for
`fotos.capibaraeventos.com`:

```nginx
server {
    server_name fotos.capibaraeventos.com;
    client_max_body_size 300m;                 # RAW uploads (see PIPELINE_MAX_UPLOAD_MB)
    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;   # required: makes session cookies Secure
        proxy_read_timeout 600s;                       # long renders / big zips
    }
}
```

Then add TLS (certbot). The `X-Forwarded-Proto: https` header is important — the app
marks session cookies `Secure` based on it.

## Updating the code

From the workstation (`/home/anderson/photorawproject`):

```bash
rsync -az --delete \
  --exclude venv --exclude data --exclude ComfyUI --exclude .git \
  --exclude web/node_modules --exclude web/dist --exclude .env \
  ./ anderson@10.200.1.1:/home/anderson/photos_capibara/
ssh anderson@10.200.1.1 'cd photos_capibara && docker compose --profile auth up -d --build'
```

The SPA is built inside the Docker image (Vite, multi-stage `Dockerfile`), so you don't
build `web/dist` locally for the server.

## Ingesting photos

Drop RAW files into `./inbox` on the host (bind-mounted to the worker at
`/data/inbox`). Editors' uploads via the UI land in `inbox/<username>/`. The worker
picks them up automatically.

## Gotchas on this box

- **Disk is tight** (host was ~91% full at deploy). Watch `df -h /`. The rembg model
  (~1.5 GB) downloads on first retouch into the `models` volume. Old scratch TIFFs are
  under the `data` volume's `work/` and are safe to delete (rebuildable from RAW).
- **Port 8080** is Nextcloud — don't reuse it. Keycloak uses **8085** here.
- Keep retouch **deterministic** (OpenCV only); generative AI is used solely for
  background replacement / object erase, and background defaults to *keep*.
- **Performance (develop + retouch are CPU-only).** rawpy/LibRaw, OpenCV (CPU build) and
  onnxruntime (CPU wheel) do not use the GPU. The dominant cost was the rembg matte:
  `birefnet-general` is ~12–22 s/photo; **`PIPELINE_REMBG_MODEL=u2net_human_seg`** (set in
  `.env`) drops it to ~0.3 s, cutting a 26 MP photo from ~41 s to ~26 s. Trade-off: u2net loses
  a little of very dark clothing (train/skirt) — fine for face retouch, weaker for
  clothing-grade / background-replace. `isnet-general-use` (~0.9 s) is a middle ground.
  After the swap the remaining time is spread across ~a dozen full-res OpenCV ops + big 16-bit
  TIFF I/O — no single hotspot.
- **AMD GPU (ROCm) is not usable for this pipeline as built.** The box has ROCm 6.3.3 + an RX
  7800 XT and `torch` works on it, but `onnxruntime-rocm` wheels don't load cleanly (ROCm 7 vs
  6.3 soname mismatches on PyPI; the 6.3 wheel throws `undefined symbol: Provider_GetHost` and
  needs system `libhipfft` that isn't installed; mixing with torch's bundled ROCm libs is
  unsafe). Even if fixed it would only accelerate the matte, which is already ~0.3 s on CPU.
  rawpy and the OpenCV filters can't use the AMD GPU without a full torch rewrite. Net: the
  model swap is the real win; GPU offload is not worth the fragility here.
- **Face retouch is scoped to the first-plane subject.** `pipeline/retouch.py` keeps only
  faces that sit on the rembg subject matte (background/second-plane guests are dropped), then
  classifies each kept face's gender so the women-only ops (lip enhancement + cheek blush /
  `skin_red`) are skipped on men. All other ops (skin smoothing, blemishes, eyes, teeth, hair)
  still apply to every foreground face.
- **Gender model.** Optional ONNX classifier at `models/gender_net.onnx` (index 0=Male,
  1=Female). If absent, every foreground face is treated as female (full retouch) — no
  regression, gating just turns off. It's the Levi-Hassner model **converted from Caffe to
  ONNX** because OpenCV 5 removed the Caffe importer. To (re)generate from the source
  `models/gender_net.caffemodel` + `models/gender_deploy.prototxt` (prototxt is committed;
  the two 45 MB binaries are git-ignored):
  ```bash
  pip install onnx==1.17.0 && pip install --no-deps caffe2onnx
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -m caffe2onnx.convert \
    --prototxt models/gender_deploy.prototxt --caffemodel models/gender_net.caffemodel \
    --onnx models/gender_net.onnx
  # then strip the no-op Dropout nodes (onnxruntime rejects their attr): see git history
  ```
  `onnx`/`caffe2onnx` are **build-time only** — the runtime loads the ONNX with `cv2.dnn`.
- **Models in Docker.** `_MODELS_DIR` is `/app/models`. The image ships the models: the
  Dockerfile `COPY models ./models` and `.dockerignore` no longer excludes `models/` (it only
  excludes the 45 MB `gender_net.caffemodel` source, which the runtime doesn't need). `deploy.sh`
  likewise rsyncs `models/` (minus the caffemodel), so the server's build context has them. Net:
  YuNet + gender classifier are in the container and face-dependent retouch works server-side.
  (The rembg matte models are separate — auto-downloaded at runtime into the `/models/rembg`
  volume via `U2NET_HOME`, not baked into the image.)
