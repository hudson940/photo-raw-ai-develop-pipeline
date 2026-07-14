# Automated RAW Photo Development Pipeline

Watch folder → SQLite queue → RAW preview extraction → AI vision analysis → RAW develop
(16-bit TIFF) → deterministic retouch → publish/archive with a decision log.

## Deploy with Docker Compose

For a server deployment, `docker compose` runs the web UI and the worker from one image,
with all durable artifacts on an S3-compatible bucket:

```bash
cp .env.example .env        # set ANTHROPIC_API_KEY, PIPELINE_WEBUI_PASSWORD, S3_* creds
docker compose up -d --build            # webui on :8765 + worker watching ./inbox
```

- **webui** — the review UI, customer share links, and render jobs (`http://host:8765`).
  Set `PIPELINE_WEBUI_PASSWORD` — the container listens on `0.0.0.0`, so without it the
  operator UI is open to the network (it warns on boot). Share links keep their own passwords.
- **worker** — watches the mounted `./inbox` for new RAWs and runs analyze → develop → retouch.
- Both share the `data` volume (SQLite queue + local file cache) and the `models` volume
  (rembg weights). Drop RAW files into `./inbox` on the host.

### Object storage (S3-compatible)

Set `PIPELINE_STORAGE=s3` and the `S3_*` variables to mirror every durable artifact — the RAW
archive, previews, final outputs, thumbnails, erase masks, and RapidRaw sidecars — to any S3
API (AWS S3, MinIO, Cloudflare R2, Backblaze B2, DigitalOcean Spaces…). Files are uploaded as
they are produced and re-downloaded on demand, so the local `data` volume is just a cache: lose
it (or start a fresh container) and the bucket repopulates it. Keys mirror the tree, optionally
under `S3_PREFIX` (e.g. `output/IMG_1.jpg` → `s3://bucket/<prefix>/output/IMG_1.jpg`). The
16-bit scratch TIFFs in `work/` are never uploaded — they are rebuildable from the RAW.

Need storage in the same stack? A **MinIO** service is included behind a compose profile:

```bash
docker compose --profile local-s3 up -d       # MinIO on :9000 (console :9001), bucket auto-created
# then in .env: S3_ENDPOINT_URL=http://minio:9000
```

With `PIPELINE_STORAGE=local` (the default) everything stays on the volume and no S3/boto3 is
used — the same code path, storage calls become no-ops.

> Design note: the pipeline's own queue/albums stay on SQLite (on a volume) — single-host compose
> doesn't need Postgres for them. Operator identity is the exception: it's delegated to Keycloak
> (below), which brings its own Postgres. Customer share links keep their built-in per-link PBKDF2
> auth. The seams are isolated (`pipeline/storage.py`, `pipeline/auth.py`, `db.connect`).

### Operator authentication & users (Keycloak)

The operator UI supports three modes, chosen by configuration:

| Config | Who can use it |
| --- | --- |
| `KEYCLOAK_URL` set | Multi-user login with roles + the in-app **Users** panel (recommended) |
| only `PIPELINE_WEBUI_PASSWORD` set | A single shared admin password (HTTP Basic) |
| neither | Open — local dev only |

With Keycloak, operators sign in through an **in-app form** (no redirect): the app exchanges the
credentials with Keycloak (direct grant) and issues a signed, stateless session cookie. Two realm
roles:

- **super_admin** — everything, plus the **Users** panel (create users, set role editor/super_admin,
  enable/disable, delete — all backed by Keycloak's Admin API).
- **editor** — the full develop toolset, but **scoped to their own photos**: an editor only sees,
  opens, redoes, erases, albums, and shares photos they own. Ownership comes from the inbox
  subfolder a RAW arrives in — `inbox/<username>/shot.CR3` belongs to `<username>`; files dropped
  in the inbox root are shared/admin-only. Super admins see every photo.

Bring up the bundled Keycloak (+ its Postgres) with the `auth` profile:

```bash
# 1. edit keycloak/realm-photoraw.json: replace CHANGE-ME-client-secret and
#    CHANGE-ME-admin-password (the initial super admin is username "admin").
# 2. set KEYCLOAK_* / SESSION_SECRET / KC_* in .env to match.
docker compose --profile auth up -d
```

The realm import creates the `photoraw` realm, the `super_admin`/`editor` roles, the confidential
`photoraw-app` client (direct grant + a service account with the `manage-users`/`view-users`/
`query-users`/`view-realm` roles the Users panel needs), and the initial `admin` super user. New
users created from the panel are ready to log in immediately (email/profile auto-completed).
Prefer an existing Keycloak? Point `KEYCLOAK_URL` at it and import the same realm file. Put HTTPS
in front in production — session cookies are marked `Secure` behind `X-Forwarded-Proto: https`.

## Setup (bare-metal / development)

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
python -m pipeline.run --once             # process all pending photos now, then exit
python -m pipeline.run --once --requeue-stuck   # also retry photos stuck in 'previewed'
python -m pipeline.status    # queue counts, failures, latest analysis
```

Pending photos are processed immediately and back-to-back; the worker only idles (the
`scan_interval` sleep) when the queue is empty. Use `--once` for a one-shot drain without
the watcher. `--requeue-stuck` resets photos left in `previewed` by an interrupted run back
to `pending`; add `--requeue-review` to also reprocess low-confidence `review` photos.

The same retouch/white-balance flags as `redo` can be passed to `run` to **force settings
on every photo** the worker processes (they override the AI's per-photo choices):

```bash
python -m pipeline.run --skin 0.3 --dewlap 0.3 --wb camera --background blur
```

Drop RAW files into `data/inbox/` (or point `PIPELINE_INBOX` at your Windows mount).
Previews land in `data/previews/`, and each photo's analysis JSON is stored on its queue row.
When a photo finishes, its RAW is moved to `data/archive/` with a **RapidRaw `.rrdata` sidecar**
next to it (`<name>.rrdata`) — the develop parameters (exposure, contrast, highlights/shadows,
saturation/vibrance, rotation) are mapped onto RapidRaw's adjustment sliders so the photo opens
in RapidRaw already developed. White balance is left at RapidRaw's as-shot (temperature/tint 0),
matching the pipeline's camera WB. `redo` rewrites it on re-render. (Only the develop/tonal params
transfer — the face/skin/hair/background retouch has no RapidRaw equivalent.)

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
| `PIPELINE_HIGHLIGHT_KNEE` | `0.75` | Highlights above this roll off smoothly toward white instead of hard-clipping (protects bright skin from blowing out); `1.0` disables |
| `PIPELINE_JPEG_QUALITY` | `90` | JPEG quality for published output images |
| `PIPELINE_KEEP_TIFFS` | _(off)_ | Keep the 16-bit `work/` TIFFs instead of deleting them |
| `PIPELINE_CANARY_MAX_TOKENS` | `1024` | Token budget for the startup vision canary (needs to be large for local *reasoning* models) |
| `PIPELINE_REASONING_EFFORT` | `low` | How hard the analysis model thinks: `off`/`low`/`medium`/`high`, or `default` to send nothing. Caps the thinking budget on the Anthropic API; sent as `reasoning_effort` to local servers. Use `default` for local GGUFs that expose no reasoning control (LM Studio logs "cannot be converted to custom KVs") |
| `PIPELINE_ANALYZE_SUBJECT_ONLY` | _(off)_ | Mask the background to gray in the analysis preview so the AI meters exposure/white-balance/color on the **subject only** (also via `--subject-only`) |
| `PIPELINE_ANALYZE_SKIN_EXPOSURE` | _(off)_ | Mask everything except **skin** in the analysis preview so the AI meters exposure on skin — avoids over-exposing skin (also via `--skin-exposure`) |

### Local models (LM Studio, Ollama, …)

The pipeline talks to any Anthropic-compatible `/v1/messages` endpoint. For a local
vision model in **LM Studio**, point the Anthropic SDK at it in `.env`:

```
ANTHROPIC_BASE_URL=http://127.0.0.1:1234
ANTHROPIC_API_KEY=sk-local          # any non-empty string
PIPELINE_MODEL=qwen3.6-vl-reap-26b-a3b   # the model id LM Studio reports
```

Reasoning models (Qwen3-VL, etc.) spend many tokens "thinking" before answering. The
startup vision canary allows for this via `PIPELINE_CANARY_MAX_TOKENS` (default 1024) — if
it were too small, the model would run out of budget mid-thought and return empty text,
which looks like a dropped image but isn't.

## Checking the parameters used for a photo

```bash
python -m pipeline.redo <id> --show    # full develop + retouch params currently stored for a photo
python -m pipeline.redo --list         # ids + state + confidence for every photo
python -m pipeline.status              # queue counts + the most recent analysis JSON
```

`--show` prints the parameters that would be used on the next render (i.e. the latest stored
analysis, including any `redo` overrides you saved). For the **history of automatic runs** —
what was actually applied each time the worker finalized a photo, with timestamps — read the
decision log:

```bash
# every recorded run for one file, newest last
grep IMG_6914 data/decision_log.jsonl | python -m json.tool   # (or | jq)
sqlite3 data/pipeline.db "SELECT id, filename, state, confidence FROM photos"
```

## Retouch (Stages 5–6)

Retouch is **deterministic image processing**, not generative AI — the photo is never
regenerated, so identity, texture, and resolution are always preserved. Operations run on
the full-resolution 16-bit TIFF (`pipeline/retouch.py`):

- **skin softening** — frequency separation: skin tone is evened and fine texture (pores/noise)
  is attenuated in proportion to `--skin` (≈−13% at 0.2, −33% at 0.5, −58% at 0.9), while
  coarser features/edges stay sharp and a floor keeps enough pore texture to avoid a plastic look
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
- **lip enhancement** (`--lips`) — restores/deepens the natural lip red plus a little richness and
  definition; lips are isolated by an adaptive test (pixels distinctly *redder* than the
  surrounding skin, inside a tight mouth ellipse), so only the lip tone is affected — teeth,
  braces and perioral skin are left alone
- **tame highlights** — recovers overexposed / shiny skin by pulling blown patches back toward
  the surrounding skin tone (forehead, nose, cheeks)
- **hair enhancement** — makes strands pop via texture + clarity (local contrast on the hair
  region), on by default for portraits; optional shimmer (raise highlights / drop shadows) and
  defrizz (trim flyaways for a smoother, straighter look)
- **clothing contrast** — CLAHE clarity on the subject's clothing (skin excluded)
- **clothing color grade** — color-aware develop of the clothing only (subject minus skin and
  hair): the AI identifies the fabric color and applies the usual moves — release the color
  (vibrance-weighted saturation), luminance, shadows, blacks, whites — with neutral fabric
  (black/white/gray) getting rich blacks / clean whites instead of saturation
- **subject exposure** — brightens (or darkens) only the subject in EV stops, background left
  untouched (uses the subject mask); useful for backlit/underexposed subjects
- **background exposure** — brightens (or darkens) only the background in EV stops, subject left
  untouched; darken a distracting/blown background so the subject stands out
- **background** — **`keep` by default, always** (the background is never changed automatically);
  change it only by passing `--background`: `auto` (inspect the scene and pick smooth/studio/keep —
  see below) / `blur` (bokeh) / `smooth` (de-wrinkle a backdrop: clone out imperfections, then
  smooth creases) / `studio` (solid backdrop — pick a color with `--studio-color`: gray/white/
  black/charcoal/blue/navy/teal/green/red/maroon/pink/purple/beige/brown or a `#hex`) / `replace`
  (generative, needs ComfyUI; falls back to `studio` when down)
  - **`--background auto`** decides for you from the background region: a neutral studio backdrop
    that fills the frame but has fold **wrinkles → `smooth`**; a backdrop that only partly fills
    the frame with **other zones visible (floor, stands, wall) → `studio`** using the backdrop's
    own tone (`white`/`gray`/`black` from its lightness); **anything else → `keep`**. Add it to
    `pipeline.run` to auto-handle every photo, or per photo via `redo <id> --background auto`.

Final images are published to `output/` as **JPEG (quality 90 by default)**, a few MB each.
The 16-bit intermediate TIFFs live in `work/` and are deleted once the JPEG is published
(set `PIPELINE_KEEP_TIFFS=1` to keep them).

**White balance** (develop stage) defaults to the camera's accurate as-shot balance
(`mode: "camera"`) — no extra correction, so colors stay true. Override per photo with a
specific color temperature via `mode: "kelvin"` (higher Kelvin = warmer; 6500 = no change),
or from the CLI with `--wb 5200 --tint 5`.

Face detection: OpenCV YuNet (`models/face_detection_yunet_2023mar.onnx`, CPU).
Subject masks: rembg with **birefnet-general** by default (much more accurate on people/clothing
than u2net — e.g. it keeps a dark gown instead of graying it out; ~15s/photo on CPU, only when a
mask op is used). Set `PIPELINE_REMBG_MODEL=u2net` for a faster, lower-quality matte. Weights
auto-download on first use.

### Re-retouch with your own choices

The AI proposes parameters; you can override them per photo and re-render in seconds:

```bash
python -m pipeline.redo --list                     # what's in the queue
python -m pipeline.redo 6 --show                   # current params for photo 6
python -m pipeline.redo 6 --reanalyze              # re-run the AI analysis, then render
python -m pipeline.redo 6 --reanalyze --subject-only   # meter exposure/color on the subject only
python -m pipeline.redo 99 --reanalyze --skin-exposure # meter exposure on skin (avoid blown skin)
python -m pipeline.redo 6 --skin 0.3 --eyes 0.2    # softer skin, brighter eye-whites
python -m pipeline.redo 6 --blemishes --skintone 0.3 --darkcircles 0.3 --iris 0.3
python -m pipeline.redo 6 --skintone 0.4 --skin-type deep   # tone to the right color for deep skin
python -m pipeline.redo 6 --skin-type olive        # force olive undertone (auto-sets --skintone)
python -m pipeline.redo 6 --skin 0.35 --skin-luminance 0.5 --skin-saturation 1.2   # bright, rich portrait skin
python -m pipeline.redo 6 --dewlap 0.3             # reduce a double chin
python -m pipeline.redo 6 --hair 0.6 --hair-shimmer 0.4 --defrizz 0.3   # hair pop
python -m pipeline.redo 6 --no-hair               # turn off default hair enhancement
python -m pipeline.redo 6 --subject-exposure 0.6  # brighten only the subject (+0.6 EV)
python -m pipeline.redo 6 --background-exposure -0.7   # darken only the background (-0.7 EV)
python -m pipeline.redo 6 --auto-levels           # per-layer auto exposure (skin/subject/bg to target)
python -m pipeline.redo 6 --skin-target 0.7 --subject-target 0.5   # tune the per-layer targets
python -m pipeline.redo 6 --shadows 25 --highlights -30   # lift shadows, recover highlights
python -m pipeline.redo 6 --wb camera              # accurate as-shot white balance (default)
python -m pipeline.redo 6 --wb 5200 --tint 5       # override white balance in Kelvin
python -m pipeline.redo 6 --background auto        # decide smooth/studio/keep from the scene
python -m pipeline.redo 6 --background blur        # bokeh background
python -m pipeline.redo 6 --background smooth      # de-wrinkle a studio backdrop
python -m pipeline.redo 6 --studio-color blue      # solid blue studio backdrop
python -m pipeline.redo 6 --tame-highlights 0.5    # recover overexposed/shiny skin
python -m pipeline.redo 6 --background replace --prompt "soft window light"
python -m pipeline.redo 6 --clothing 0.4 --intensity subtle --from-raw
python -m pipeline.redo 6 --cloth-color red --cloth-pop 0.35 --cloth-shadows -0.15   # pop a red dress
python -m pipeline.redo 6 --cloth-color black --cloth-blacks -0.25   # deep, rich blacks
python -m pipeline.redo 6 --cloth-color white --cloth-whites -0.15   # keep whites clean, no clipping
python -m pipeline.redo 6 --erase-mask 6.png                  # remove the white regions of the mask
python -m pipeline.redo 6 --erase-method generative --erase-prompt "empty lawn"   # Generative Fill
python -m pipeline.redo 6 --clear-erase                       # forget the saved erase mask
```

Overrides are saved back to the queue DB, so they stick for future re-renders. `--reanalyze`
re-runs the AI vision analysis first (useful after changing the model or prompt); you can
combine it with overrides, e.g. `redo 6 --reanalyze --skin 0.3`.

**Inspect the skin selection.** `python -m pipeline.redo <id> --dump-mask` writes a side-by-side
`[original | overlay]` JPEG to `data/output/_mask_<name>_<id>.jpg` where selected skin is tinted
red (brighter = stronger mask). Use it to see exactly what the skin ops act on — face, perioral
skin, nose, and body skin (arms/shoulders) — and to spot any holes that would read as a different
tone. **Colour tone correction covers the whole face and body skin — eyes and lips included** —
so skin colour is even right across the face with no differently-toned patches around the eyes or
mouth. (Skin *softening* and *blemish removal* still skip the eyes and lips via a separate
feature mask, so those features are never blurred or cloned.) The face mask also fills small
colour holes (nose specular, skin next to the lips) inside the face geometry. **Hair is rejected**
from the body-skin path even when it's close to skin colour, because hair is less saturated (lower
chroma) and more textured (strands) than skin — so brown hair on the shoulders isn't recoloured.

**Skin-tone correction is skin-type aware and targets skin _hue_.** `skin_tone_correction`
(`--skintone`) no longer applies one fixed warm push to every face. It classifies the subject's
skin type (via **ITA°**, the dermatology-standard Individual Typology Angle, or the type the AI
reports) and then **gently nudges each skin pixel toward the natural, healthy hue (~55-58°) for
that type** — keeping the tone the camera captured rather than restyling it, while pulling colour
casts back onto the skin line. The rotation is **per-pixel and asymmetric**: green/yellow patches
(common in shadowed skin, where a uniform rotation would leave green blotches) are pulled toward
skin hue, while the red side moves only gently so natural blush survives. Chroma is then deepened
by `--skin-saturation` (default `1.15`) so skin looks rich rather than flat/washed-out, and
`--skin-luminance` (default `0.45`) **brightens the skin** — with extra lift on the lips/cheeks —
the way portrait editors use the Lightroom Color-Mixer Orange/Red luminance sliders. Skin stays
natural by default; for a deliberately warmer/orange look raise `--skin-warmth` (default `0`).
A `--skin-red` flush (default `0.35`) adds healthy red to the skin midtones/cheeks — restoring the
warm red the camera's colour science shows that a flat RAW develop loses — and the red side of the
tone correction is left untouched so lips and cheek blush keep their colour. These knobs
(`--skintone` strength, `--skin-saturation` richness, `--skin-luminance` brightness, `--skin-warmth`
warmth, `--skin-red` flush, plus `--lips`) mirror the classic quinceañera-portrait skin edit — skin
smoothing is separate (`--skin`), and the "green primary" balance is handled by the built-in
per-pixel green-cast removal. It now covers
**body skin too** (arms, shoulders, chest), not just the face — skin-colored pixels on the
subject matte are corrected along with the face, so exposed skin no longer stays a different
(yellower) tone than the face. Controls:
`--skin-type fair|light|medium|olive|tan|brown|deep` (auto by default) and `--skin-warmth -1..1`
(bias toward orange `+` / yellow `-`; default `0`, set globally with `PIPELINE_SKIN_WARMTH`).
Note white balance matters most here: an inaccurate `--wb` (e.g. an AI Kelvin override) cools and
desaturates skin — prefer `--wb camera` so tone correction works on the camera's accurate colour.

**Per-layer auto exposure (`--auto-levels`).** Instead of one global exposure for the whole
frame, this meters each region — **face skin**, the **rest of the subject** (body/clothing/hair),
and the **background** — separately and corrects each toward its own midtone target through the
masks the retouch stage already builds. The regions are disjoint, so lifting a dim face never
brightens the background, and if the skin still has blown pixels after metering it is pulled
back down automatically (a one-flag fix for over/under-exposed skin). It runs on the already
developed TIFF (no re-develop), so it's fast to iterate. Targets default to skin 0.72 /
subject 0.5 / background off; tune per photo with `--skin-target`, `--subject-target`,
`--background-target`, or globally via `PIPELINE_AUTO_*` env vars. Corrections are clamped to
±1.25 EV per layer (`PIPELINE_AUTO_MAX_EV`) so metering can never overcook.

**Erase objects (Content-Aware / Generative Fill).** Paint a mask over anything that should
disappear — a cable, a bystander, lint on a backdrop — and the pipeline removes it and fills the
hole from its surroundings. Two fill engines: `content-aware` (default; deterministic OpenCV
inpainting, fast, ideal for small distractions) and `generative` (ComfyUI diffusion inpaint —
Photoshop's Generative Fill — better for large objects; needs ComfyUI running and only
regenerates the masked area). The erase runs **before every other retouch op**, so the subject
matte, skin masks and exposure metering all see the cleaned frame. Masks live in `data/masks/`
(white = remove) and are saved with the photo's parameters, so later re-renders keep the removal
until you `--clear-erase`. The web UI (below) draws these masks with a brush directly on the photo.

To re-analyze in bulk through the queue, requeue with a fresh analysis and drain:

```bash
python -m pipeline.run --once --requeue-stuck --reanalyze   # re-analyze stuck photos
```

## Quality gate (auto-reject bad shots)

Before spending AI analysis + develop on a photo, a deterministic OpenCV check runs on its
preview and flags shots not worth processing — **blurry** (variance-of-Laplacian focus),
**underexposed** (dark + crushed shadows), **overexposed** (bright + blown highlights). Flagged
photos are parked in a **`rejected`** state, **default to not-selected**, and skip the expensive
stages. They still appear in the gallery (dimmed, with a ⚠ badge and a `rejected` filter); an
operator can **Process anyway** from the lightbox to force one through, and when a rejected photo
is added to an album it starts as *discarded*.

Tune or disable via env: `PIPELINE_QUALITY_GATE=0` turns it off; `PIPELINE_QUALITY_BLUR_MIN`,
`PIPELINE_QUALITY_DARK_MAX`, `PIPELINE_QUALITY_BRIGHT_MIN`, `PIPELINE_QUALITY_CLIP_FRAC` set the
thresholds (defaults are tuned for the ~1536px preview).

## Web UI (review & batch redo)

```bash
python -m pipeline.webui                 # http://127.0.0.1:8765 (localhost only)
python -m pipeline.webui --host 0.0.0.0  # reachable from the LAN
```

A single-page gallery over the queue DB — no new dependencies (stdlib HTTP server + one HTML
file). It shows every photo with its latest render (state, confidence, portrait flag), with
filtering by state and filename search.

- **Lightbox** — click a photo: full-size render, compare with the **original preview**, inspect
  the exact develop/retouch parameters and the copy-paste `redo` command that reproduces them.
  **Zoom** with the +/−/Fit buttons, mouse wheel, drag-to-pan or double-click; **Redo this…**
  opens the wizard for just that photo.
- **Crop** — drag a rectangle (with optional locked aspect ratio: 1:1, 4:5, 5:7, 3:2, 2:3, 16:9)
  over the full-frame original and apply; it re-develops the RAW with the new crop.
- **Erase objects** — paint over parts of the photo with a brush (right in the lightbox), pick
  **Content-Aware Fill** (fast, deterministic) or **Generative Fill** (ComfyUI, optional prompt),
  and re-render. The mask is saved per photo and can be deleted again from the same toolbar.
- **Batch redo wizard** — select photos (shift-click for ranges), then *Redo selected…* opens a
  step-by-step wizard (Skin → Face → Hair & clothing → Light & develop → Background → Review)
  with every CLI parameter as a slider/selector. Only the parameters you explicitly enable are
  overridden — everything else keeps each photo's current values. Single-photo selections
  pre-fill the photo's current parameters. The review step shows the equivalent
  `python -m pipeline.redo` command, plus *from RAW*, *re-run AI analysis* and the analysis
  metering options (*subject only*, *skin exposure*).
- **Jobs drawer** — renders run sequentially in the background through the same code path as
  `pipeline.redo` (identical output names, sidecars, DB bookkeeping) with per-photo progress,
  failure reasons, and cancel. The gallery refreshes thumbnails as renders finish.

The JSON API behind it (`/api/photos`, `/api/photos/<id>`, `/api/redo`, `/api/jobs`,
`/api/photos/<id>/erase`, `/thumb/<id>`, `/img/<id>`) is plain HTTP — scriptable with `curl`.

### Albums & customer share links

Group photos into **albums** and send a customer a single link to proof them:

1. Select photos in the gallery → **Albums…** → create an album (or add to an existing one).
2. In the album row, set a **link password**, pick a **permission**, and *Create share link*:
   - **select & discard only** — the customer sees just the album, with big ✓ *Select* /
     ✗ *Discard* buttons on every photo and in the lightbox (plus zoom and the original-preview
     compare). Nothing else: no redo, no develop tools, no other photos.
   - **full develop** — additionally the whole develop toolset on the album's photos: the redo
     wizard, crop, erase objects, and a jobs drawer showing *their own* render jobs only.
     Re-running AI analysis stays operator-only (it spends API credits) and photos outside the
     album are rejected server-side.
3. Copy the link (`https://…/share/<token>`) and send it with the password. The link uses
   **HTTP Basic auth** — the customer's browser asks for the password (any username); it is
   checked against a per-link PBKDF2 hash stored in the DB. Revoke a link at any time from the
   same dialog.

Customer decisions land live in the operator UI: pick the album in the header dropdown to see
✓/✗ badges on each photo and filter by *selected / discarded / undecided* (it auto-refreshes
every 10 s). Counts also appear in the Albums dialog.

To make links reachable, run with `--host 0.0.0.0` (or reverse-proxy). When binding beyond
localhost, protect the operator UI too: `--admin-password ...` (or `PIPELINE_WEBUI_PASSWORD`)
puts the whole operator surface behind HTTP Basic auth — share links keep their own passwords.
For anything crossing the open internet, terminate TLS in front (e.g. Caddy/nginx): Basic auth
sends the password base64-encoded, so it needs HTTPS to be private.
