<div dir="rtl">

# تحويل الصوت إلى نص

</div>

Local Arabic speech-to-text. Upload an MP3 (or WAV/M4A/OGG/FLAC/WebM/MP4…), watch the
Arabic transcript stream in right-to-left as it is produced, and export it as TXT, SRT or
VTT. FastAPI backend, [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for the
model, no cloud service and no API keys — after the first model download it works offline.

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH`
- Optional: an NVIDIA GPU. Without one it runs on CPU, just slower.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Optional, for NVIDIA GPUs — CTranslate2 loads cuDNN/cuBLAS from these wheels:
.venv/bin/pip install -r requirements-gpu.txt
```

## Run

```bash
./run.sh                 # http://127.0.0.1:8000
S2T_PORT=8100 ./run.sh   # if port 8000 is taken
```

`run.sh` puts the pip-installed CUDA libraries on `LD_LIBRARY_PATH`, so GPU execution works
without a system-wide CUDA install. The first transcription downloads the selected model
(~1.6 GB for the default) and the UI shows a "preparing the model" step while it does.

Open the page, drop in a file, pick a model, press **ابدأ التحويل**.

## Models

Choose per request in the UI. The picker names the trade-off rather than the checkpoint
(«سريع ودقيق», «أبطأ وأدق», «خفيف — يعمل على المعالج»); the keys below are what the API and
`S2T_MODEL` expect. Each is downloaded once and then cached.

| Model | UI label | Params | Download | When to use |
| --- | --- | --- | --- | --- |
| `large-v3-turbo` *(default)* | سريع ودقيق | 809 M | ~1.6 GB | Best speed/accuracy balance. Runs `float16` on GPU. |
| `large-v3` | أبطأ وأدق | 1550 M | ~3.1 GB | Hardest audio — heavy dialect, noise, poor recordings. Quantized to `int8_float16` so it fits a 4 GB card. |
| `small` | خفيف — يعمل على المعالج | 244 M | ~0.5 GB | Low-resource machines or a quick smoke test. Noticeably less accurate. |

Only one model stays resident in VRAM — switching models on the GPU unloads the previous
one, because two large checkpoints will not fit in 4 GB.

### Measured on an RTX 3050 (4 GB), 399 s of Arabic podcast audio

| Model | Wall clock | Peak VRAM | Segments |
| --- | --- | --- | --- |
| `large-v3-turbo` | ~30 s (≈13× realtime) | ~2.0 GB | 103 |
| `large-v3` | ~50 s (≈8× realtime) | ~3.7 GB of 4.0 GB | 87 |

Both returned language probability 1.0. `large-v3` was the more accurate of the two on this
clip — for example it produced «وتصل إلى **الطلاقة** في اللغة» where turbo produced
«الطلاق». On a 4 GB card `large-v3` leaves little headroom, so close other GPU workloads
before using it.

### What makes the Arabic output accurate

These are not library defaults; they are set in `app/transcriber.py`:

- **Language pinned to `ar`.** Whisper's auto-detection misfires on short clips and on
  Quranic recitation. Auto-detect is still available in the dropdown.
- **`condition_on_previous_text=False`.** The single most effective guard against Whisper's
  repetition loops, which Arabic triggers far more often than English.
- **Beam search, `beam_size=5`**, rather than greedy decoding.
- **Silero VAD** trims silence, so the model is not asked to invent text to fill it.
- **Temperature fallback ladder** `0.0 → 1.0` with compression-ratio and log-probability
  thresholds, so a degenerate decode is retried instead of returned.

## Configuration

All optional, all read at startup.

| Variable | Default | Meaning |
| --- | --- | --- |
| `S2T_MODEL` | `large-v3-turbo` | Model preselected in the UI |
| `S2T_DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `S2T_COMPUTE_TYPE` | `auto` | Any CTranslate2 compute type, e.g. `int8`, `float16` |
| `S2T_BEAM_SIZE` | `5` | Beam width; `1` is greedy and faster |
| `S2T_CPU_THREADS` | `0` | `0` lets CTranslate2 decide |
| `S2T_MAX_UPLOAD_MB` | `500` | Upload size cap |
| `S2T_JOB_TTL_SECONDS` | `3600` | How long a finished transcript stays retrievable |
| `S2T_WORK_DIR` | `./work` | Scratch space for uploads (files are deleted after each job) |
| `S2T_MODEL_DIR` | `./models` | Where model weights are cached; set to `""` to use the shared Hugging Face cache |
| `S2T_HOST` / `S2T_PORT` | `127.0.0.1` / `8000` | Bind address, read by `run.sh` |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/transcribe` | Multipart `file`, `model`, `language`, `task` → `202 {job_id, …}` |
| `GET` | `/api/jobs/{id}` | Full state: status, progress, segments, joined text |
| `GET` | `/api/jobs/{id}/stream` | SSE — `snapshot`, `status`, `segment`, `done`, `error` |
| `GET` | `/api/jobs/{id}/download?fmt=txt\|srt\|vtt` | Transcript as a file |
| `DELETE` | `/api/jobs/{id}` | Cancel and forget a job |
| `GET` | `/api/models`, `/api/health` | Metadata for the UI; runtime and device info |

```bash
curl -X POST http://127.0.0.1:8000/api/transcribe \
  -F "file=@audio.mp3" -F "model=large-v3-turbo" -F "language=ar"
# {"job_id":"…","status":"queued",…}

curl -N http://127.0.0.1:8000/api/jobs/<job_id>/stream
curl "http://127.0.0.1:8000/api/jobs/<job_id>/download?fmt=srt" -o audio.srt
```

Transcription is a background job rather than a blocking request, so a long recording
cannot time out the upload, and text appears while the rest is still being decoded. Only
one decode runs at a time — a 4 GB GPU cannot hold two large models.

## Right-to-left output

- The page is `dir="rtl" lang="ar"` and the layout uses logical CSS properties only, so it
  mirrors correctly rather than being flipped by hand.
- Timestamps are isolated with `dir="ltr"` and `unicode-bidi: isolate`. Without that, a
  Latin-digit `00:01:23` next to Arabic text gets visually reordered by the bidi algorithm.
- Exported TXT lines and SRT/VTT cues are prefixed with U+200F (RIGHT-TO-LEFT MARK), so
  subtitle players and text editors align them correctly even when a line starts with a
  digit or a Latin word.

## Project layout

```
app/
  config.py       settings + the model registry
  audio.py        ffprobe validation, ffmpeg → 16 kHz mono WAV
  transcriber.py  model cache, Arabic decoding options, CUDA→CPU fallback
  jobs.py         job records, SSE fan-out, TTL purge
  worker.py       the pipeline: convert → load → decode → publish
  formats.py      segments → TXT / SRT / VTT
  main.py         routes
  static/         index.html, styles.css, app.js (no build step)
  static/fonts/   vendored IBM Plex Sans Arabic + Noto Naskh Arabic (OFL)
tests/            pytest suite
docs/superpowers/specs/   design document
```

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest                      # fast: no model download, no GPU needed
```

The suite injects a fake transcriber, but runs real `ffmpeg` against generated audio,
because that is where format bugs actually live. To also exercise a real model end to end:

```bash
S2T_RUN_SLOW=1 S2T_TEST_AUDIO=/path/to/arabic.mp3 .venv/bin/pytest -m slow -s
```

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Badge says CPU although you have an NVIDIA GPU | The cuDNN/cuBLAS wheels are missing — `pip install -r requirements-gpu.txt` — or you started uvicorn directly instead of via `run.sh`, which sets `LD_LIBRARY_PATH`. Details are in `/api/health` under `runtime.warnings`. |
| "address already in use" | Something else owns the port. `S2T_PORT=8100 ./run.sh`. |
| Job fails with a CUDA out-of-memory detail | Pick `large-v3-turbo` or `small`, or force `S2T_DEVICE=cpu`. GPU failures already fall back to CPU automatically at load time. |
| "الأداة ffmpeg غير مثبّتة" | Install ffmpeg; both `ffmpeg` and `ffprobe` must be on `PATH`. |
| Transcript repeats the same phrase | Usually very noisy or near-silent audio. Try `large-v3`, which is more robust on hard input. |

## License

Uses Whisper weights (MIT) via faster-whisper (MIT).
