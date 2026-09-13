# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build description for the Windows desktop build.

A .spec file is a Python script executed by PyInstaller, not a config file, so the
collection below is computed rather than hand-listed.

Two rules this file exists to enforce:

* The build environment installs ``requirements.txt`` only. Installing
  ``requirements-gpu.txt`` would drag ~1.5 GB of CUDA libraries into the bundle
  and defeat the on-demand download in ``app/cuda_setup.py``.
* ``upx=False``. UPX compression markedly raises antivirus false-positive rates on
  PyInstaller output, and buys nothing we need.
"""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(SPECPATH).parent  # noqa: F821 - SPECPATH is injected by PyInstaller

# -- data ------------------------------------------------------------------
# The frontend, found at runtime by main.py's Path(__file__).parent / "static".
datas = [(str(ROOT / "app" / "static"), "app/static")]

# faster-whisper ships the Silero VAD models (silero_encoder_v5.onnx,
# silero_decoder_v5.onnx) as package data. transcriber.py runs with
# vad_filter=True, so without these the first transcription fails.
datas += collect_data_files("faster_whisper")
datas += collect_data_files("tokenizers")

# -- binaries --------------------------------------------------------------
binaries = []
for package in ("ctranslate2", "onnxruntime", "av", "tokenizers"):
    binaries += collect_dynamic_libs(package)

# ffmpeg and ffprobe, put on PATH by launcher.configure_environment().
bin_dir = ROOT / "packaging" / "bin"
for tool in ("ffmpeg.exe", "ffprobe.exe"):
    source = bin_dir / tool
    if not source.is_file():
        raise SystemExit(
            f"{source} is missing. Run packaging/build.ps1, which fetches a static "
            "ffmpeg build, or drop the two binaries in by hand."
        )
    binaries.append((str(source), "bin"))

# -- imports ---------------------------------------------------------------
# uvicorn selects its protocol, loop and lifespan implementations dynamically, so
# static analysis cannot see them.
hiddenimports = collect_submodules("uvicorn")
hiddenimports += [
    "app.main",
    "app.cuda_setup",
    "anyio._backends._asyncio",
    "encodings.idna",
]

excludes = [
    "tkinter",
    "matplotlib",
    "pytest",
    "_pytest",
    "setuptools",
    "uvloop",      # POSIX only; uvicorn falls back to asyncio on Windows
    "watchfiles",  # --reload only, never used in the packaged app
]

a = Analysis(  # noqa: F821
    [str(ROOT / "app" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="speech2text",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # the console window is also the user's quit control
    disable_windowed_traceback=False,
    icon=str(ROOT / "packaging" / "app.ico")
    if (ROOT / "packaging" / "app.ico").is_file()
    else None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="speech2text",
)
