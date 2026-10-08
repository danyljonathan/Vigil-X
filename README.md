# VIGIL X

### Conversational Multi-Camera Video Intelligence

**See Everything. Ask Anything.** · *Ask. Search. Verify. See.*

HackNex 2026 · Problem statement **HNX26EPS05 — Multi-Stream Video Intelligence with Conversational Query**

---

## What is VIGIL X?

VIGIL X is an AI investigator for **recorded** camera footage. You record videos (phones work fine), register each
one against a camera (CAM-01, CAM-02, …) with its real recording start time, and VIGIL X processes the actual video:

* detects objects with **Ultralytics YOLO11m**,
* follows them with **anonymous ByteTrack / BoT-SORT tracks** (`PERSON-TRACK-001`, `CAR-TRACK-004`, …),
* classifies **day / low-light / night** lighting and enhances *analysis copies* of dark frames,
* generates **deterministic events** (person entered/exited view, vehicle detected, person with backpack, …),
* extracts **actual evidence frames** and **short clips** from the original video,
* indexes frames and object crops with **SigLIP 2** image–text embeddings in **FAISS**,

and then lets you **ask questions in plain language** — "*Did a red car pass through the main gate?*",
"*What about the staircase?*", "*Did the same person later appear at Lab 1?*" — with every answer backed by a
real evidence frame and a link that opens the original video at the exact moment.

### The no-hallucination rule

The database and the video evidence are the source of truth — **never** the language model.

```
USER QUESTION → QUERY PARSER → STRUCTURED QUERY → DATABASE / FAISS SEARCH → DETECTIONS / TRACKS / EVENTS
            → ACTUAL EVIDENCE FRAME / CLIP → (optional VLM verification) → GROUNDED ANSWER
```

* If nothing reliable is found, VIGIL X answers **"No verified evidence was found in the uploaded footage."**
* If the footage is too dark for the requested detail: **"Insufficient visual evidence to reliably identify the requested detail."**
* Optional LLM summaries are rejected by a **grounding check** if they mention any time, camera, track id or
  percentage that is not present in the retrieved evidence.
* Semantic (SigLIP) matches that are not confirmed by the detector, colour analysis or a VLM are labelled
  **POSSIBLE · UNVERIFIED**.
* Cross-camera continuity is reported as **VERIFIED / LIKELY / UNCERTAIN / NOT VERIFIED**; without a VLM the
  maximum is LIKELY, and it is always described as appearance consistency — never identity.

---

## Features

| Area | What you get |
|---|---|
| Ingestion | Website upload (drag & drop, real byte progress) **or** drop files in `data/videos/` → **SCAN VIDEOS** → confirm camera/date/start time/timezone → **PROCESS ALL** |
| Formats | MP4 · MOV · AVI · MKV (FFmpeg/OpenCV). Non-H.264 files get a derived browser playback copy; originals are never modified |
| Duplicates | SHA-256 of every file; re-imports are refused with *"This video has already been imported."* Reprocessing only on explicit request |
| Timestamps | Every detection/event/evidence has a **video-relative** time (`00:02:31`) and an **absolute** time (`2026-10-08 10:02:31 +05:30`) = recording start + offset |
| Processing | Background worker, real progress %, cancel, restart-safe, GPU (CUDA) with CPU fallback |
| Detection | YOLO11m (configurable) — person, bicycle, car, motorcycle, bus, truck, backpack, handbag, suitcase (only classes the model supports) |
| Tracking | ByteTrack (default) or BoT-SORT; anonymous IDs; entry/exit edges; motion direction |
| Lighting | DAY / LOW_LIGHT / NIGHT / UNKNOWN per frame; CLAHE + gamma + denoise on analysis copies only |
| Events | PERSON_DETECTED, PERSON_ENTERED, PERSON_EXITED, VEHICLE_DETECTED, BICYCLE_DETECTED, PERSON_WITH_BACKPACK, PERSON_WITH_BAG, MULTIPLE_PEOPLE, OBJECT_MOVEMENT, CROSS_CAMERA_CONTINUITY |
| Evidence | Raw frame (original pixels), annotated frame (bounding boxes), enhanced copy for dark footage, crop, short H.264 clip |
| Search | Two-stage: SQL filters + detector evidence + SigLIP/FAISS open-vocabulary search → optional Qwen2.5-VL (or other VLM) verification |
| Conversation | Follow-ups ("What about the staircase?", "Only at night.", "Show the second one."), stored conversations & query history |
| Clarify once | Unknown places ("main gate") trigger one question; the answer is stored in SQLite and survives restarts |
| Cross-camera | Cautious continuity chains (CAM-01 → CAM-02 → CAM-03): distinctiveness z-score, margin, mutual best match, colour signature and time gap |
| UI | Dashboard · Cameras · Videos · Events · Search · Conversations · Data Explorer · Settings (React + Vite + Tailwind) |
| Export | CSV / JSON of any query's exact results, or of the event table |

---

## Architecture

```
                         VIGIL X
                            │
                    ┌───────▼───────┐
                    │ User Question │
                    └───────┬───────┘
                    ┌───────▼───────┐   deterministic parser (+ optional LLM, validated)
                    │ Query Parser  │
                    └───────┬───────┘
             ┌──────────────┼──────────────┐
        Time Filter    Camera Memory   Semantic Query
             └──────────────┼──────────────┘
                    ┌───────▼───────┐   SQLite: tracks / events / detections
                    │ FAISS + SigLIP│   FAISS: frame + object-crop vectors
                    └───────┬───────┘
                     Top candidates
                    ┌───────▼───────┐
                    │  VLM verify   │   Qwen2.5-VL via Ollama / OpenAI-compatible / local, or Claude vision
                    └───────┬───────┘
                    ┌───────▼───────┐
                    │ Evidence rank │
                    └───────┬───────┘
                    ┌───────▼───────┐
                    │ Grounded answer│  template answer (+ optional grounding-checked LLM summary)
                    └───────┬───────┘
                    ┌───────▼───────┐
                    │  VIGIL X UI   │
                    └───────────────┘
```

**Processing pipeline (per video):** probe → playback copy (if needed) → sampled frames @ `TRACKING_FPS` →
lighting → (enhancement) → YOLO (batched) → ByteTrack → index frames @ `TARGET_FPS` saved to `data/frames` →
detections → SigLIP embeddings (frames + crops) → track aggregation → events → evidence frames + clips →
FAISS update → cross-camera candidates.

---

## Project structure

```
VIGIL-X/  (this folder)
├── backend/
│   ├── main.py              FastAPI app (also serves the built UI)
│   ├── config.py            settings from .env
│   ├── database.py          SQLite engine (WAL), init + default cameras
│   ├── models/              SQLAlchemy ORM (cameras, videos, frames, detections, tracks, events, evidence, …)
│   ├── schemas/             Pydantic request schemas + serializers
│   ├── routes/              cameras, videos, events, evidence, search, chat, dashboard, explorer, system
│   ├── services/            detector, tracking, lighting, embedding, vector_search, query_parser, search,
│   │                        event, evidence, llm, vlm, memory, cross_camera, video, device
│   ├── utils/               time, files (safe paths, hashing), colours, media (FFmpeg)
│   ├── workers/             processing queue + pipeline
│   └── requirements.txt
├── frontend/                React + Vite + Tailwind UI
├── data/
│   ├── videos/              ORIGINAL footage (never modified)
│   ├── processed/           playback copies, FAISS indexes
│   ├── frames/ evidence/ clips/ thumbnails/
│   └── vigil_x.db           SQLite database
├── database/                db_tools.py (stats, integrity, backup, schema)
├── models/                  YOLO weights + SigLIP (downloaded)
├── scripts/                 setup/start scripts, model + sample downloaders, parallel downloader
├── tests/                   unit, API and end-to-end tests
├── logs/                    vigil_x.log
├── .env.example
└── README.md
```

---

## Requirements

* Windows 10/11 (Linux/macOS also work with the equivalent commands)
* **Python 3.10 – 3.12** (3.12 recommended; PyTorch/FAISS wheels are not yet available for 3.14)
* **Node.js 20+** for the frontend (or let the setup script install a project-local Node via `pip install nodejs-wheel`)
* NVIDIA GPU optional (CUDA is auto-detected; CPU works, just slower)
* FFmpeg: bundled automatically via `imageio-ffmpeg` (a system FFmpeg on PATH is used if present)
* ~6 GB disk for dependencies + models

---

## Windows installation

### 1. Backend

```powershell
cd backend
py -3.12 -m venv venv            # or: uv venv --seed venv --python 3.12
venv\Scripts\activate
python -m pip install --upgrade pip
# NVIDIA GPU (recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# (CPU only: pip install torch torchvision)
pip install -r requirements.txt
cd ..
copy .env.example .env
```

Or in one step: `powershell -ExecutionPolicy Bypass -File scripts\setup_backend.ps1`

### 2. Models

```powershell
backend\venv\Scripts\python scripts\download_models.py
```

This stores `models/yolo11m.pt` and `models/siglip2-base-patch16-224/` inside the project so VIGIL X runs offline.
(`--yolo yolo11n` gives a smaller, faster detector.) If you skip this step, YOLO weights are fetched automatically on
first processing and SigLIP from Hugging Face on first use.

### 3. Frontend

```powershell
cd frontend
npm install
npm run build        # production build served by the backend at http://localhost:8000
npm run dev          # OR hot-reload dev server at http://localhost:5173 (proxies /api to :8000)
```

No Node.js? `backend\venv\Scripts\pip install nodejs-wheel` then use `backend\venv\Scripts\npm.exe`, or run
`powershell -ExecutionPolicy Bypass -File scripts\setup_frontend.ps1`.

---

## Environment variables (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `./data` | all footage + derived data |
| `DATABASE_URL` | `sqlite:///./data/vigil_x.db` | SQLite file |
| `MODEL_PATH` | `models/yolo11m.pt` | YOLO weights (name or path) |
| `DETECT_CLASSES` | person,bicycle,car,… | classes to keep |
| `TRACKER` | `bytetrack` | or `botsort` |
| `TRACKING_FPS` | `5` | detector + tracker rate |
| `TARGET_FPS` | `1` | index-frame rate (saved + embedded) |
| `PROCESS_EVERY_N_FRAMES` | — | overrides `TARGET_FPS` |
| `EMBEDDING_MODEL` | `google/siglip2-base-patch16-224` | SigLIP / SigLIP 2 model |
| `LOW_LIGHT_ENHANCEMENT` | `true` | enhance analysis copies of dark frames |
| `LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY` | none | `anthropic`, `openai` (compatible), `ollama` |
| `VLM_PROVIDER` / `VLM_MODEL` / `VLM_API_KEY` | none | `ollama` (e.g. `qwen2.5vl:7b`), `openai` (e.g. DashScope `qwen-vl-max`), `anthropic`, `local` (`Qwen/Qwen2.5-VL-3B-Instruct`) |
| `TOP_K` | `30` | FAISS candidates |
| `MAX_UPLOAD_SIZE_MB` | `2048` | upload limit |
| `GENERATE_CLIPS` | `true` | cut event clips during processing |
| `DEVICE` | `auto` | `auto`, `cuda`, `cpu` |
| `DEFAULT_TIMEZONE` | `Asia/Kolkata` | default for cameras/videos |
| `CORS_ORIGINS` | `http://localhost:5173` | dev server origins |

Secrets are only read from `.env` / environment and are never displayed in the UI.

**LLM/VLM are optional.** Without them VIGIL X uses its deterministic parser and template answers, and verifies with
real CV/database evidence only. When `LLM_PROVIDER=anthropic` is set, the default model is `claude-opus-5-5`; requests
include Anthropic's server-side refusal fallback (`fallbacks="default"`), and fall back to a plain request if the
installed SDK does not support it.

---

## Running

```powershell
# backend (+ built UI) — from the project root
backend\venv\Scripts\python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
#   or: scripts\start_backend.ps1      or: cd backend; venv\Scripts\python main.py
```

Open **http://localhost:8000** (built UI) — or http://localhost:5173 when using `npm run dev`. API docs: `/docs`.

---

## Adding, scanning and processing videos

**Method A — upload:** *Videos → Upload video* → drop a file → choose camera, recording date, start time, timezone →
*Upload & register* (tick *Start processing* to queue it immediately).

**Method B — folder:** copy files into `data/videos/`, e.g. `CAM-01.mp4`, `CAM-02.mp4`, `CAM-03.mp4` →
*Videos → Scan videos* → VIGIL X lists `New videos found: 3` (SHA-256 hashed, duplicates flagged, camera suggested
from the filename) → confirm camera + date + start time + timezone → *Register & process* (or *PROCESS ALL* later).

New footage added later (e.g. `CAM-01-night.mp4`) just needs **Scan → Confirm → Process** — no rebuild.

Processing shows real progress: reading video, preparing playback copy, extracting frames, detecting objects,
tracking, generating embeddings, events, evidence, updating the search index, cross-camera analysis.

> **Recording start time matters.** Absolute timestamps are `start + video offset`. Phone files often carry a
> creation time; VIGIL X pre-fills it as a suggestion but you must confirm it. If you fix it later, every derived
> absolute timestamp is re-stamped automatically.

---

## Searching footage

Open **Search** and ask, for example:

* *Show all people detected at the CTC entrance.* → *What about the staircase?* → *Only at night.* → *Show the second one.*
* *Did a red car pass through the main gate?* · *Which camera saw the white car?* · *Find a person wearing a black shirt.*
* *Show me people carrying bags.* · *Was there a bicycle?* · *What happened between 10 and 10:30?* · *What happened last night?*
* *Show everything at CAM-01.* · *Show me the last person who entered the building.*
* *What did the person look like?* · *Did the same person later appear at Lab 1?* · *Did someone move from CAM-01 to CAM-03?*

Each result card shows object/event, camera, location, recording date, absolute + video timestamps, confidence
(the detector confidence of the detection in the evidence frame), lighting, description, the **evidence basis**
(detector, colour analysis, bag association, semantic similarity, VLM) and the **actual evidence frame**, with
**VIEW FRAME · PLAY CLIP · OPEN VIDEO** (opens the original at that timestamp). Results can be sorted by time,
camera, confidence or object, and exported as CSV/JSON.

### Evidence workflow

```
query → result → track/event/detection → evidence (frame #, video time, absolute time)
      → raw frame (original pixels) / annotated frame / clip → original video (SHA-256 recorded)
```

`GET /api/evidence/{id}` returns the full chain, including the original file path and its SHA-256.

### Camera alias memory (clarify once)

Ask about a place VIGIL X does not know (*main gate*, *parking area*, *reception*) and it asks once:
*"Which camera or region corresponds to the main gate?"* Pick a camera; the mapping (`main gate → CAM-01`) is stored
in SQLite (`camera_aliases`) and is used from then on — including after a backend restart. Manage aliases in
**Settings** or via `POST /api/cameras/alias`.

### How open-vocabulary matching is decided (calibrated)

SigLIP's raw scores are not usable as absolute probabilities on CCTV crops, so VIGIL X uses rules calibrated on real
footage:

* **Object attributes** ("red car", "person wearing a black shirt"): FAISS retrieves the candidates' crop vectors,
  then a **zero-shot contrast** checks that the requested colour beats every other colour for that object
  (on real crops this agreed with the pixel colour analysis). A result is **MATCHED** only when pixel colour analysis
  *and* the SigLIP check agree; if only one agrees it is shown as **POSSIBLE · UNVERIFIED**. Colours from night
  footage or tiny crops are never treated as verified.
* **Things the detector cannot detect** ("dog", "umbrella"): whole-frame search must beat generic scene prompts by a
  margin (absent concepts stay at or below zero), and results are always labelled unverified unless a VLM confirms.

### Cross-camera continuity

For a source track (from your previous results, or found from the question), VIGIL X looks at tracks of the same
object type on other cameras within `CONTINUITY_MAX_GAP_MINUTES` and follows the chain hop by hop
(CAM-01 → CAM-02 → CAM-03). Calibration showed that SigLIP makes *everyone* look similar (different people ≈ 0.90–0.92
cosine, the same track ≈ 0.96), so a raw similarity threshold would invent matches. Instead a candidate is **LIKELY**
only if it is **distinctive** (z-score ≥ 2.5 against the source's similarity to all other tracks), **unambiguous**
(margin over the runner-up), the **mutual best match**, **colour-consistent** and **time-plausible**. Partial evidence
gives **UNCERTAIN**; otherwise **NOT VERIFIED** ("Cross-camera continuity could not be verified."). **VERIFIED** requires
a configured VLM to independently agree. It is appearance consistency of anonymous tracks — never identity.

### Night / low-light

Every sampled frame gets a lighting class. Dark frames are enhanced (CLAHE + gamma + bilateral denoise) **only in
derived analysis copies** used for detection and shown as an "enhanced (analysis copy)" evidence view. Colour
attributes are never treated as verified in night footage; such queries answer with the "insufficient visual
evidence" message instead of guessing.

---

## API (summary)

```
GET  /api/health                 GET  /api/dashboard              GET  /api/settings
GET  /api/cameras                POST /api/cameras                PUT/DELETE /api/cameras/{id}
POST /api/cameras/alias          GET  /api/cameras/aliases
GET  /api/videos                 GET  /api/videos/{id}            POST /api/videos/upload
POST /api/videos/scan            POST /api/videos/register        POST /api/videos/process
GET  /api/videos/{id}/status     GET  /api/videos/{id}/timeline   GET  /api/videos/{id}/stream  (HTTP Range)
GET  /api/detections             GET  /api/tracks                 GET  /api/events
GET  /api/evidence/{id}          GET  /api/evidence/{id}/frame    GET  /api/evidence/{id}/clip
POST /api/search                 POST /api/chat                   GET  /api/conversations[/{id}]
GET  /api/export?format=csv|json&query_id=…                       GET  /api/explorer/{table}
```

---

## Testing

```powershell
backend\venv\Scripts\python scripts\fetch_sample_videos.py   # small public real sample videos -> tests/fixtures
backend\venv\Scripts\python -m pytest                        # or: scripts\run_tests.ps1
```

Tests run in an isolated temporary data directory (your `data/` is never touched) and cover: video registration,
SHA-256 duplicates, timestamp maths, upload validation, path traversal, query parsing (all demo queries),
camera/object/time filtering, multiple results, no-result behaviour, conversation memory, alias persistence across
sessions, low-light handling, cross-camera caution, API endpoints, and an **end-to-end run on real sample footage**
that checks every evidence frame against the original video's pixels.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip` cannot find torch | Use Python 3.10–3.12, not 3.14 |
| `No Python at '...AppData\Roaming\uv\python\...'` | The venv's base Python was installed from a sandboxed app (e.g. the Claude desktop app), which redirects `AppData`. This project ships its base Python in `backend\python312`; make sure `backend\venv\pyvenv.cfg` says `home = D:\VigilX\backend\python312` |
| Very slow downloads on campus Wi-Fi | Networks that throttle each connection: use `python scripts/fast_download.py URL OUT` (parallel ranges) and `python scripts/prefetch_wheels.py pinned.txt wheels/` then `pip install --no-index --find-links wheels -r pinned.txt` |
| "Backend offline" in the UI | Start the backend (port 8000) |
| Video plays but cannot seek | Use the app's player (served with HTTP Range); for HEVC/AVI/MKV a playback copy is created during processing |
| "Semantic search unavailable" | Run `scripts/download_models.py`; detector-based search still works |
| Processing is slow | Lower `TRACKING_FPS`, use `MODEL_PATH=yolo11n.pt`, or use a CUDA GPU |
| Wrong absolute times | Edit the video's recording start time (Videos → Edit); everything is re-stamped |
| Video stuck after a crash | On restart it is marked FAILED with a clear message; click Process |

Logs: `logs/vigil_x.log`. Database tools: `python database/db_tools.py stats|integrity|backup|schema`.

---

## Hackathon demo (2 minutes)

1. **Search:** *Did a red car pass through the main gate?* → first time: VIGIL X asks which camera is the main gate → pick CAM-01.
2. Ask again → **YES/NO** with camera, location, video + absolute time, confidence and the actual evidence frame → *Play clip* / *Open video*.
3. Restart the backend, ask again → no clarification (alias persisted).
4. **Open vocabulary:** *Show me people carrying bags.* → multiple evidence cards with their evidence basis.
5. **Follow-ups:** *Show all people at the entrance.* → *What about the staircase?* → *Show the second one.* → *What did the person look like?*
6. **Cross-camera:** *Did the same person later appear at Lab 1?* → chain with LIKELY / UNCERTAIN / NOT VERIFIED — never faked.

Record CAM-01 (entrance), CAM-02 (staircase), CAM-03 (lab) with fixed phone positions, wide views and consenting
volunteers walking the route; include a night/low-light clip.

---

## Privacy

VIGIL X uses anonymous tracking and evidence-based video analysis. It does not identify individuals by name.

* No facial recognition, no face embeddings, no identity matching.
* People are anonymous tracks (`PERSON-TRACK-007`); descriptions are limited to visible clothing colours and carried objects.
* No age, gender, ethnicity, occupation or intent inference — gendered words in queries are treated as "person".
* All processing is local; LLM/VLM providers are opt-in.
* Use consenting volunteers for demonstrations.
