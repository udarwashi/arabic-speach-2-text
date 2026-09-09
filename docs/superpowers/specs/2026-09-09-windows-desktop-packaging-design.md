# Windows desktop packaging — design

**Date:** 2026-09-09
**Status:** approved, ready for implementation planning

## Goal

Deliver this app to a non-technical Windows user as one file they double-click. No Python,
no ffmpeg, no terminal, no venv. It must run fast on an NVIDIA GPU when the machine has
one and fall back to CPU when it does not, without the user knowing either word.

## Non-goals

- macOS or Linux builds. Windows only.
- Mobile. Rejected: no in-browser or on-device runtime preserves the Arabic accuracy this
  app was tuned for.
- Rewriting the frontend into a UI framework. The existing HTML/CSS/JS in `app/static/`
  ships unchanged.
- Auto-update. Out of scope; a new installer is handed over manually.
- Code signing. Out of scope for now; see *Known rough edges*.

## Decisions taken

| Decision | Choice | Why |
| --- | --- | --- |
| Architecture | Local uvicorn on `127.0.0.1` + the user's default browser | Zero UI rewrite; the frontend and server already exist |
| Bundler | PyInstaller, `--onedir` | `--onefile` re-extracts ~500 MB to temp on every launch (15–20 s startup) |
| Distribution | Inno Setup → one `speech2text-setup.exe`, per-user install | Per-user avoids the UAC prompt entirely |
| ffmpeg | `ffmpeg.exe` + `ffprobe.exe` bundled, exposed via `PATH` | `app/audio.py` already resolves both through `PATH`; no code change |
| CUDA runtime | Downloaded from PyPI on first run, **only if an NVIDIA driver is present** | Bundling costs ~1.08 GB for a user who may have no GPU |
| Password | Disabled | Single-user local app; `.env` is absent so `S2T_PASSWORD` is already empty |
| Data location | `%LOCALAPPDATA%\speech2text\` | `Program Files` is not writable without admin |

## Architecture

Two new modules. **No changes to `config.py`, `audio.py`, `transcriber.py`, `main.py`,
`jobs.py`, `worker.py`, `formats.py` or `auth.py`.** The launcher configures the process
through environment variables and the DLL search path before any of those are imported.

```
speech2text.exe  (PyInstaller bootloader)
   └── app/launcher.py                 <- new, the frozen entry point
         ├── app/cuda_setup.py         <- new, GPU probe + CUDA provisioning
         └── app/main.py               <- unchanged, served by uvicorn
```

### `app/launcher.py`

Runs, strictly in this order:

1. **Resolve the data directory.** `%LOCALAPPDATA%\speech2text\` with `models\`, `work\`
   and `logs\` beneath it, created if absent.
2. **Set `S2T_MODEL_DIR` and `S2T_WORK_DIR`** to those paths. This must happen before
   anything imports `app.config`, because `get_settings()` is `@lru_cache`d — the first
   call fixes configuration for the process lifetime. Without it, `config.PROJECT_ROOT`
   (derived from `__file__`) points inside the PyInstaller bundle and the 1.6 GB model
   would be re-downloaded and discarded on every launch.
3. **Prepend the bundled binary directory to `os.environ["PATH"]`** so `audio.py`'s
   `shutil.which("ffmpeg")` and its bare `"ffprobe"` invocation find the shipped copies.
   `PATH` is the right mechanism here specifically because ffmpeg is launched as a
   subprocess.
4. **Configure file logging** to `logs\speech2text.log`, rotating, so a failure on the
   user's machine is diagnosable without a terminal.
5. **Provision CUDA** via `cuda_setup.ensure_cuda()` — see below. Never fatal.
6. **Claim a free port** by binding `127.0.0.1:0` and reading the assigned port back.
   Removes "port 8000 is taken" as a failure mode.
7. **Start uvicorn on a daemon thread** against `app.main:app`.
8. **Poll the server until it answers**, then `webbrowser.open` the localhost URL. Opening
   the browser before the server is listening produces a connection-refused page.
9. **Block on the console window**, showing an Arabic message explaining that closing this
   window stops the app. This window is the user's quit control.

Path resolution helper: `sys._MEIPASS` when `sys.frozen` is set, else the repo root, so
`python -m app.launcher` works unfrozen for development and testing.

### `app/cuda_setup.py`

```
ensure_cuda() -> bool          # True if the CUDA DLL directory is registered
  has_nvidia_driver() -> bool  # ctypes.WinDLL("nvcuda.dll") succeeds
  download_and_extract()       # pinned wheels -> %LOCALAPPDATA%\speech2text\cuda\
```

- **Detection** loads `nvcuda.dll`, which the NVIDIA display driver installs and nothing
  else does. On non-Windows it returns `False` immediately, so the module imports cleanly
  on Linux for the test suite.
- **Provisioning**, only when a driver is found and the cache directory is empty:
  downloads two pinned wheels over HTTPS, verifies SHA-256 before opening either, and
  extracts the DLLs from `nvidia/*/bin/` into the cache directory.

  | Wheel | Size | SHA-256 |
  | --- | --- | --- |
  | `nvidia_cudnn_cu12-9.1.1.17-py3-none-win_amd64.whl` | 680 MB | `d7c4b96b1c5ca8e4dc0bdbf2ce386903ba03faa94d2997b72769fa5048cb5f1f` |
  | `nvidia_cublas_cu12-12.4.5.8-py3-none-win_amd64.whl` | 397 MB | `5a796786da89203a0657eda402bcdcec6180254a8ac22d72213abc42069522dc` |

  Versions match `requirements-gpu.txt`, so the GPU path is the one already tested on the
  RTX 3050. Download progress is printed to the console; a partial download is written to
  a `.part` file and renamed only on a hash match, so an interrupted first run retries
  cleanly rather than caching corruption.
- **Registration** calls `os.add_dll_directory()` on the cache directory *and* prepends it
  to `PATH`. Both are needed: since Python 3.8 extension modules no longer search `PATH`
  for their dependent DLLs, but CTranslate2 opens the CUDA libraries by name at run time,
  and a plain `LoadLibrary` searches `PATH` while ignoring directories added the other way.
- **`cuda_runtime_available()`** then tries to load `cublas64_12.dll` and `cudnn64_9.dll`,
  and the launcher pins `S2T_DEVICE=cpu` when they will not load. This is not belt and
  braces; it closes a real gap found by running real speech through the packaged build:

  > `RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`

  CTranslate2 reports a CUDA device whenever the *driver* can see the card, so
  `resolve_device()` picks `cuda` even with no cuBLAS present. The existing fallback in
  `transcriber.py:146-168` wraps only model *loading*; this failure happens later, during
  decoding, where nothing catches it. A user whose CUDA download had failed would have got
  an error rather than a slower answer.
- **Downloads resume.** A `Range` request continues an interrupted `.part` file rather than
  restarting 1.1 GB. The pinned SHA-256 still covers the whole file, so a stale or corrupt
  partial simply fails the check and is discarded.
- **Console writes go through `say()`.** The Windows console defaults to a code page that
  cannot encode Arabic, and an unguarded `print` there raises `UnicodeEncodeError` —
  which once aborted the entire download over a cosmetic message.
- **Failure is never fatal.** No driver, no network, a hash mismatch, or a read-only disk
  all log a warning and return `False`. `transcriber.resolve_device()` then reports `cpu`
  on its own.
- **Escape hatch:** `S2T_SKIP_CUDA=1` skips the whole path, for testing the CPU route on a
  GPU machine. The launcher also pins `S2T_DEVICE=cpu` in that case: CTranslate2 still
  reports a CUDA device without the cuDNN DLLs, because it sees the card through the
  driver, so otherwise the flag would only reach the CPU by way of a failed model load.

### Why no transcription logic changes

`app/transcriber.py:114` already resolves `auto` → `cuda` when
`ctranslate2.get_cuda_device_count() > 0`, and lines 146–168 already catch a failed CUDA
model load and retry on `cpu`/`int8`. "Fast GPU by default, CPU when there is no GPU" is
existing, tested behaviour. Packaging only has to put the DLLs where they can be found.

### `packaging/`

| File | Purpose |
| --- | --- |
| `speech2text.spec` | PyInstaller spec — `datas`, `binaries`, `hiddenimports`, `upx=False` |
| `build.ps1` | Windows build script: venv, deps, PyInstaller, Inno Setup |
| `installer.iss` | Inno Setup script, `PrivilegesRequired=lowest` |
| `bin/` | `ffmpeg.exe`, `ffprobe.exe` — gitignored, fetched by `build.ps1` |

The spec must collect, at minimum:

- `app/static/**` → `app/static` (so `main.py:44`'s `Path(__file__).parent / "static"` resolves)
- `faster_whisper` data files — the Silero VAD models `silero_encoder_v5.onnx` and
  `silero_decoder_v5.onnx`, ~1.2 MB. Missing, `vad_filter=True` fails or tries to download.
- `onnxruntime` native libraries
- `ctranslate2` native libraries
- uvicorn's dynamically imported submodules (`uvicorn.protocols.*`, `uvicorn.loops.*`,
  `uvicorn.lifespan.*`) as `hiddenimports`
- `ffmpeg.exe`, `ffprobe.exe` from `packaging/bin/`

The build venv installs `requirements.txt` **only** — never `requirements-gpu.txt`, or the
1.5 GB of CUDA libraries land in the bundle and defeat the on-demand design.

## Data layout on the user's machine

```
%LOCALAPPDATA%\Programs\speech2text\    <- installed by Inno Setup
    speech2text.exe
    _internal\...                        (DLLs, static assets, ffmpeg)

%LOCALAPPDATA%\speech2text\              <- created by the launcher
    models\      Whisper weights, ~1.6 GB after first transcription
    cuda\        cuDNN + cuBLAS DLLs, ~1.5 GB, only on NVIDIA machines
    work\        scratch audio, deleted per job by existing code
    logs\        speech2text.log
```

The uninstaller removes the program directory. It leaves `%LOCALAPPDATA%\speech2text\`
in place, since re-downloading 1.6 GB after an accidental uninstall is worse than leaving
a cache behind; the installer offers removing it as an unchecked option.

## Error handling

| Situation | Behaviour |
| --- | --- |
| No NVIDIA driver | Skip CUDA silently, run on CPU |
| CUDA download fails or hash mismatch | Log, delete the partial file, run on CPU |
| CUDA runtime not loadable | `S2T_DEVICE=cpu` pinned before the server starts |
| CUDA present but model load fails | Existing fallback in `transcriber.py:146-168` retries on CPU |
| Console cannot encode Arabic | `say()` degrades to the console's own encoding; never raises |
| Port unavailable | Cannot happen — the OS assigns the port |
| Browser fails to open | The console prints the URL for manual entry |
| Any startup exception | Logged to `logs\speech2text.log`, printed, console held open so the message is readable |

## Testing

Unit tests run on Linux under the existing `pytest` setup and must not require Windows:

- `tests/test_launcher.py` — data-dir resolution frozen vs unfrozen (monkeypatching
  `sys.frozen`/`sys._MEIPASS`), env vars set before settings load, `PATH` prepending,
  free-port selection returns a bindable port.
- `tests/test_cuda_setup.py` — `has_nvidia_driver()` returns `False` off Windows; a mocked
  `WinDLL` drives both branches; extraction against a fixture zip pulls only the DLLs;
  a deliberate hash mismatch leaves the cache empty and returns `False`; `S2T_SKIP_CUDA=1`
  short-circuits.

The existing suite must keep passing untouched — that is the check that the launcher
changed nothing in the app.

Not unit-testable, and therefore a manual checklist run on Windows before handover:
built exe launches; browser opens; a real MP3 transcribes end to end; TXT/SRT/VTT export;
second launch is fast and re-downloads nothing; on a GPU machine the log records `cuda`;
with `S2T_SKIP_CUDA=1` it records `cpu` and still completes.

## Measured outcome

Built on 2026-09-09 with Windows Python 3.11.6, PyInstaller 6.22.2 and Inno Setup 6.

| | |
| --- | --- |
| `dist\speech2text\` (onedir bundle) | 436 MB, of which ffmpeg + ffprobe are 197 MB |
| `speech2text-setup.exe` | **116 MB** |
| Silent install | exit 0, no UAC prompt, per-user under `%LOCALAPPDATA%\Programs\speech2text` |
| Start Menu entry | `تحويل الصوت إلى نص.lnk`, correct Arabic |
| CUDA bundled | none — confirmed absent from `_internal\` |
| CUDA provisioned on first run | 11 DLLs, 1.5 GB, both wheels passed their checksums |
| Transcription | verified on both `cuda`/`float16` and `cpu`/`int8`, identical output |

## Known rough edges

- **SmartScreen.** The unsigned installer triggers "Windows protected your PC". The user
  must click *More info → Run anyway*. Tell them in advance. Azure Trusted Signing (~$10
  per month) is the cheap fix if this becomes a recurring handover.
- **First run on a GPU machine downloads ~1.08 GB before the browser opens**, with only a
  console progress line. Acceptable once; revisit if it proves confusing.
- **Antivirus false positives** are possible with any PyInstaller build. `upx=False` and
  `--onedir` both reduce the risk.
- **The console window** is unpolished but is also the only quit control. A tray icon
  (`pystray`) or a real window (`pywebview`) is the upgrade path if it grates.
- **Build machine.** PyInstaller cannot cross-compile; the build runs on the Windows side
  of this WSL2 host, or on a GitHub Actions `windows-latest` runner.

## Background reading

`docs/desktop-app-primer.md` explains the desktop-packaging concepts this design assumes —
what a PE binary is, how PyInstaller's analysis works, `sys._MEIPASS`, DLL search order,
installers, and code signing.
