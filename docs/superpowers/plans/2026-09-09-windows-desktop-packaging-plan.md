# Windows desktop packaging — implementation plan

**Design:** `docs/superpowers/specs/2026-09-09-windows-desktop-packaging-design.md`
**Date:** 2026-09-09

## Build environment (verified)

| Fact | Value |
| --- | --- |
| Dev box | WSL2 Linux, interop **enabled** — Windows binaries run from bash |
| Windows Python | 3.11.6, `py -3.11` |
| winget | available |
| Inno Setup | installing via winget |
| ffmpeg | not on Windows PATH — static build fetched into `packaging/bin/` |
| NVIDIA driver | `C:\Windows\System32\nvcuda.dll` **exists** — GPU path is testable here |
| Free space on C: | 94.5 GB |

**Constraint:** `cmd.exe`/PowerShell cannot use a `\\wsl.localhost\...` UNC path as a working
directory. Every build step runs from a native Windows path; the tree is synced to
`C:\build\speech2text` first.

## Task breakdown

### T1 — `app/cuda_setup.py` (TDD)

Write `tests/test_cuda_setup.py` first; it must pass on Linux.

Public surface:
```python
def has_nvidia_driver() -> bool
def cuda_dir() -> Path
def ensure_cuda(log: Logger) -> bool
```

- `has_nvidia_driver()` returns `False` immediately when `sys.platform != "win32"`, else
  `ctypes.WinDLL("nvcuda.dll")` inside try/except.
- `ensure_cuda()` short-circuits on `S2T_SKIP_CUDA=1`; returns `False` when no driver;
  registers the dir with `os.add_dll_directory` and returns `True` when the cache is
  already populated; otherwise downloads, verifies, extracts, registers.
- Download: stream to `<name>.part`, hash while streaming, rename only on match, delete on
  mismatch. Console progress (percentage of known content-length).
- Extract: only members matching `nvidia/*/bin/*.dll`, flattened into `cuda_dir()`.
- Wheels pinned in a module-level table with the SHA-256 values from the design doc.
- **Never raises.** Every failure path logs and returns `False`.

Tests: off-Windows detection; mocked `WinDLL` for both branches; `S2T_SKIP_CUDA=1`
short-circuit; extraction from a fixture zip pulls only DLLs; hash mismatch leaves the
cache empty and returns `False`; an already-populated cache skips downloading.

**Verify:** `pytest tests/test_cuda_setup.py -v`

### T2 — `app/launcher.py` (TDD)

Write `tests/test_launcher.py` first.

Public surface:
```python
def bundle_dir() -> Path          # sys._MEIPASS when frozen, else repo root
def data_dir() -> Path            # %LOCALAPPDATA%\speech2text, XDG-ish fallback off Windows
def configure_environment() -> Path
def free_port() -> int
def main() -> int
```

Ordering inside `main()` is load-bearing and must be asserted by a test:
`configure_environment()` sets `S2T_MODEL_DIR`/`S2T_WORK_DIR` **before** `app.config` is
imported, because `get_settings()` is `@lru_cache`d.

- `configure_environment()` creates `models/`, `work/`, `logs/`, `cuda/`; sets the two env
  vars; prepends `bundle_dir()/bin` to `os.environ["PATH"]`.
- `free_port()` binds `127.0.0.1:0`, reads the port, closes.
- `main()` sets up rotating file logging to `logs/speech2text.log`, calls `ensure_cuda`,
  starts uvicorn on a daemon thread, polls `/` until it answers (bounded, ~30 s), opens the
  browser, prints the Arabic quit message, blocks until interrupted.
- Any startup exception is logged and printed, and the console is held open so the user can
  read it.

Tests: frozen vs unfrozen `bundle_dir()`; `data_dir()` honours a monkeypatched
`LOCALAPPDATA`; env vars set and dirs created; `PATH` prepended; `free_port()` returns a
bindable port; the env-before-config-import ordering.

**Verify:** `pytest tests/test_launcher.py -v` then the full suite.

### T3 — `packaging/speech2text.spec`

PyInstaller onedir, `upx=False`, `console=True`, entry `app/launcher.py`.
`datas`: `app/static` → `app/static`; `collect_data_files("faster_whisper")` for the Silero
VAD `.onnx` models. `binaries`: `collect_dynamic_libs` for `onnxruntime` and `ctranslate2`;
`packaging/bin/ffmpeg.exe` and `ffprobe.exe` → `bin/`.
`hiddenimports`: `collect_submodules("uvicorn")` plus `app.main`, `app.cuda_setup`,
`encodings.idna`. `excludes`: `tkinter`, `matplotlib`, `pytest`.

**Verify:** built by T5.

### T4 — `packaging/installer.iss`

Inno Setup: `PrivilegesRequired=lowest`, `DefaultDirName={localappdata}\Programs\speech2text`,
Arabic + English display name, Start Menu shortcut, uninstaller, unchecked task offering to
delete `%LOCALAPPDATA%\speech2text` (the model/CUDA cache) on uninstall.
Output: `dist\speech2text-setup.exe`.

**Verify:** built by T5.

### T5 — `packaging/build.ps1`

Sync repo → `C:\build\speech2text` (excluding `.git`, `.venv`, `models`, `work`, `dist`,
`build`, `__pycache__`); create `.venv-win` with `py -3.11`; `pip install -r requirements.txt
pyinstaller` (**never** `requirements-gpu.txt`); assert `packaging/bin/ffmpeg.exe` and
`ffprobe.exe` exist; run PyInstaller against the spec; run `ISCC.exe` against the `.iss`;
copy `speech2text-setup.exe` back to the repo's `dist/`.

**Verify:** run it; `dist/speech2text-setup.exe` exists and is a plausible size (~350–450 MB).

### T6 — `.gitignore`

Add `packaging/bin/`, `build/`, `dist/`, `.venv-win/`.

**Verify:** `git status` clean of build artefacts.

### T7 — Windows smoke test (manual, scripted where possible)

1. Run the installer; confirm no UAC prompt.
2. Launch from the Start Menu; browser opens; page renders in Arabic RTL.
3. On this GPU box, `logs/speech2text.log` records the CUDA download then `device=cuda`.
4. Transcribe a real MP3 end to end; export TXT, SRT, VTT.
5. Close, relaunch: starts in ~2 s, downloads nothing.
6. `S2T_SKIP_CUDA=1`: the launcher pins `S2T_DEVICE=cpu`, the log records `device=cpu`, and transcription still completes.
7. Uninstall leaves `%LOCALAPPDATA%\speech2text` unless the option was ticked.

**Verify:** all seven observed; record results in the final report.

## Execution order

T1 → T2 (both TDD, Linux) → T3, T4, T6 (config files) → T5 (build) → T7 (smoke test).
T3/T4/T6 are independent of each other and can be written in one pass.

## Out of scope

Code signing, auto-update, macOS/Linux builds, tray icon, `pywebview` window.
