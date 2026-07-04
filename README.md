# Automated RAW Photo Development Pipeline

Watch folder → SQLite queue → RAW preview extraction → AI vision analysis → RAW develop
(16-bit TIFF) → deterministic retouch → publish/archive with a decision log.

## Setup

Already done on this machine: a `venv/` with all Python deps, and exiftool 13.59 installed
to `~/.local/bin` (no root needed). To recreate elsewhere:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt              # includes rawpy — decodes RAWs without system tools
export ANTHROPIC_API_KEY=sk-ant-...          # or `ant auth login`
```

Preview extraction tries, in order: **exiftool** (embedded JPEG, fastest) → **rawpy**
(direct RAW decode, pure pip) → **darktable-cli** (optional; only needed from Stage 4 on).
`sudo apt install exiftool darktable` is the nicer setup when root is available, but not required.

## Run

```bash
python -m pipeline.run       # starts watcher + worker; Ctrl-C to stop
python -m pipeline.status    # queue counts, failures, latest analysis
```

Drop RAW files into `data/inbox/` (or point `PIPELINE_INBOX` at your Windows mount).
Previews land in `data/previews/`, and each photo's analysis JSON is stored on its queue row.

## Flow

```
inbox/ ──watcher──▶ pending ──▶ previewed ──▶ analyzed        (ready for Stage 4)
                       │ (exiftool /            │
                       │  darktable-cli)        ├─▶ review    (low AI confidence)
                       └────────────────────────┴─▶ failed    (after 3 attempts)
```

- **Watcher** uses polling (works on `/mnt/c` and Samba shares) and waits for files to finish
  copying before enqueueing.
- **Preview** falls back from exiftool → rawpy → darktable-cli, so it works even without
  any system packages installed.
- **Analysis** uses Claude with structured outputs — the API guarantees the response matches
  the `PhotoAnalysis` schema in `pipeline/analysis.py`, so no fragile JSON parsing.
- **Confidence gate**: analyses below `PIPELINE_CONFIDENCE_THRESHOLD` (default 0.6) go to
  `review` instead of proceeding — nothing low-confidence flows to automatic processing.
- Failures retry with exponential backoff (30s, 60s, 120s), then park in `failed` with the
  error recorded. Rate limits requeue without burning an attempt.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `PIPELINE_ROOT` | `./data` | Base directory for inbox/previews/output/archive/db |
| `PIPELINE_INBOX` | `$PIPELINE_ROOT/inbox` | Watch folder (set to your Windows mount) |
| `PIPELINE_MODEL` | `claude-opus-4-8` | Vision model for analysis |
| `PIPELINE_CONFIDENCE_THRESHOLD` | `0.6` | Below this → human review |
| `PIPELINE_PREVIEW_LONG_EDGE` | `1536` | Preview size sent to the AI |
| `PIPELINE_MAX_ATTEMPTS` | `3` | Retries before a photo is marked failed |
| `PIPELINE_JPEG_QUALITY` | `90` | JPEG quality for published output images |
| `PIPELINE_KEEP_TIFFS` | _(off)_ | Keep the 16-bit `work/` TIFFs instead of deleting them |

## Inspecting an analysis

```bash
sqlite3 data/pipeline.db "SELECT filename, state, confidence FROM photos"
python -m pipeline.status   # pretty-prints the most recent analysis JSON
```

## Retouch (Stages 5–6)

Retouch is **deterministic image processing**, not generative AI — the photo is never
regenerated, so identity, texture, and resolution are always preserved. Operations run on
the full-resolution 16-bit TIFF (`pipeline/retouch.py`):

- **skin softening** — frequency separation: skin tone is evened while pores/texture stay
- **blemish removal** — erases pimples, acne marks, skin tags and stray hairs *over skin* by
  cloning nearby skin (`cv2.inpaint`); blob-selective + safety-capped so wrinkles/moles/
  identity are never touched
- **skin tone correction** — evens blotchy chroma and restores a healthy tone (texture kept)
- **dark circles** — lightens under-eye circles / eye bags toward surrounding skin brightness
- **dewlap reduction** — subtly lifts a double chin / sagging under-chin skin (a gentle,
  feathered liquify push, capped so it stays natural)
- **eye brightening** — brightens the whites (sclera) and reduces bloodshot redness
- **iris enhance** — subtle saturation/clarity pop on the irises only
- **teeth whitening** — desaturates yellow + brightens inside the detected mouth
- **clothing contrast** — CLAHE clarity on the subject's clothing (skin excluded)
- **background** — `keep` / `blur` (bokeh) / `smooth` (de-wrinkle a backdrop: clone out
  imperfections, then smooth creases) / `studio` (neutral backdrop) / `replace` (generative,
  needs ComfyUI; falls back to `studio` when ComfyUI is down)

Final images are published to `output/` as **JPEG (quality 90 by default)**, a few MB each.
The 16-bit intermediate TIFFs live in `work/` and are deleted once the JPEG is published
(set `PIPELINE_KEEP_TIFFS=1` to keep them).

**White balance** (develop stage) defaults to the camera's accurate as-shot balance
(`mode: "camera"`) — no extra correction, so colors stay true. Override per photo with a
specific color temperature via `mode: "kelvin"` (higher Kelvin = warmer; 6500 = no change),
or from the CLI with `--wb 5200 --tint 5`.

Face detection: OpenCV YuNet (`models/face_detection_yunet_2023mar.onnx`, CPU).
Subject masks: rembg / U²-Net (weights auto-download to `~/.u2net` on first use).

### Re-retouch with your own choices

The AI proposes parameters; you can override them per photo and re-render in seconds:

```bash
python -m pipeline.redo --list                     # what's in the queue
python -m pipeline.redo 6 --show                   # current params for photo 6
python -m pipeline.redo 6 --skin 0.3 --eyes 0.2    # softer skin, brighter eye-whites
python -m pipeline.redo 6 --blemishes --skintone 0.3 --darkcircles 0.3 --iris 0.3
python -m pipeline.redo 6 --dewlap 0.3             # reduce a double chin
python -m pipeline.redo 6 --wb camera              # accurate as-shot white balance (default)
python -m pipeline.redo 6 --wb 5200 --tint 5       # override white balance in Kelvin
python -m pipeline.redo 6 --background blur        # bokeh background
python -m pipeline.redo 6 --background smooth      # de-wrinkle a studio backdrop
python -m pipeline.redo 6 --background replace --prompt "soft window light"
python -m pipeline.redo 6 --clothing 0.4 --intensity subtle --from-raw
```

Overrides are saved back to the queue DB, so they stick for future re-renders.
