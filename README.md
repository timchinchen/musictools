# MusicTools - Rock MP3 to Guitar Tab (Starter)

This is a basic starter application for uploading an MP3 and getting an initial
"rock-oriented" guitar output:

- Chord progression
- Riff note suggestions
- Simple tab-like mapping (string/fret pairs)

Analysis uses librosa (tempo, key, chords), onset + pitch tracking for riff
notes, 16th-note quantization on the beat grid, and a global DP mapper with a
user-selectable riff preset. A fallback path remains if decoding fails.

## Project Structure

- `backend/` - FastAPI service and upload analysis endpoint
- `frontend/` - static web UI for file upload and result display

## Quick Start

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

3. Run the API server:

```bash
uvicorn backend.app.main:app --reload
```

4. Open the frontend in your browser:

- Option A: directly open `frontend/index.html`
- Option B: serve it locally using a static server

By default, the frontend calls `http://127.0.0.1:8000`.

## AWS deployment (simple)

This stack is a **long-lived Python process** with **CPU-heavy** analysis (librosa, NumPy, SciPy). A small **EC2** or **Lightsail** VM is the straightforward option. **AWS Lambda** is a poor fit for multi‑MB uploads and multi‑second analysis jobs.

1. **Instance** — Ubuntu LTS or Amazon Linux 2023; **t3.small** or larger is a reasonable starting point. Open inbound **80** and **443** (or **8000** only while testing).
2. **App** — Clone the repo on the instance, create a virtual environment, then `pip install -r backend/requirements.txt` from the project root. If wheels fail, you may need build tools (for example `cmake`); see your platform’s Python packaging notes.
3. **Run** — From the **repository root**:

   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
   ```

   Keep this running with **systemd** or another process supervisor (ad‑hoc: **tmux** / **screen**). The API serves the UI at `/` (redirects to `/frontend/index.html`) and under `/frontend/`.
4. **HTTPS (recommended)** — Put **nginx**, **Caddy**, or an **Application Load Balancer** in front and reverse‑proxy to `http://127.0.0.1:8000`.
5. **Frontend API URL** — In `frontend/index.html`, set `API_BASE` to your public site origin (for example `https://your-domain.com`). If the browser loads the UI from the same host and port as the API, you can use `window.location.origin` instead so you do not hardcode a URL.
6. **Persistence** — Uploaded MP3s are stored under `backend/uploads/`, and upload history SQLite lives under `backend/data/`. Ensure the instance volume is large enough and backed up if you need uploads to survive redeploys.

**Health check:** `GET /api/health` returns `{"status":"ok"}`.

## API

`POST /api/upload`

- Form-data field: `file` (MP3 file)
- Returns: JSON containing `file_id`, `filename`, `size_bytes`

`POST /api/analyze?file_id=<uploaded-id>&riff_preset=<preset>`

- Analyzes a previously uploaded MP3 by `file_id`
- Optional `riff_preset`: `low_string_rock` | `mid_neck_rock` | `lead_solo` (default: `mid_neck_rock`)
- Returns: JSON including `tempo_bpm`, `key`, `chords`, `riff_notes`,
  `tab_suggestions`, `riff_preset`, and `debug` metadata.

## Next Steps

- Add job queue for larger files.
- Add editor UI for manually correcting tab suggestions.
