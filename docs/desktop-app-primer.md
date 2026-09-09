# From web app to desktop app — a primer for web developers

You know how to build and ship a web app. This document explains what changes when the
thing has to run on someone else's computer with no server involved, what an `.exe`
actually is, which frameworks exist and why you would pick one, and exactly how this
project — a Python/FastAPI speech-to-text app — becomes a file a non-technical user
double-clicks.

Nothing here assumes prior desktop experience. It does assume you know what a server,
a process, and a dependency are.

---

## 1. The one idea everything else follows from

In web development, responsibility is split cleanly:

| Concern | Who handles it |
| --- | --- |
| Runtime (Python, Node) | You, on your server |
| Dependencies | You, on your server |
| Rendering the UI | The user's browser |
| Updates | You deploy; the user reloads |
| Where files are written | Your server's filesystem |

On the desktop, **all of it moves onto the user's machine**. You are no longer shipping
a URL, you are shipping *the entire runtime environment*. Your user does not have Python.
They will not install ffmpeg. They will never open a terminal. Whatever your program
needs on the day it runs must be inside the thing you handed them, or it must be
something the program fetches for itself.

Every desktop packaging tool that exists is an answer to three questions:

1. **Where does the UI render?** (a native window? a bundled browser? the OS's webview? the user's own browser?)
2. **Where does the language runtime come from?** (bundled? assumed installed? compiled away?)
3. **How does the user start and stop it?** (a shortcut, a window's X button, a tray icon)

Pick an answer to each and you have chosen your architecture.

---

## 2. What an `.exe` actually is

An `.exe` on Windows is a file in **PE format** (Portable Executable). Internally it has:

- a **header** describing the target CPU architecture and where execution starts,
- **sections** of machine code and constant data,
- an **import table** listing DLLs it needs and which functions from each,
- optional embedded **resources** (icon, version info, manifests).

When you double-click it, the Windows loader maps it into memory, walks the import table
loading each DLL, and jumps to the entry point. That is the whole mechanism.

**A DLL** (Dynamic Link Library) is the same format without an entry point — a library
loaded at runtime rather than baked in at build time. `kernel32.dll`, `python312.dll` and
`cudnn_ops64_9.dll` are all DLLs. The Linux equivalent is `.so`, macOS `.dylib`.

### Compiled languages vs interpreted languages

This distinction determines what packaging even means:

- **Go, Rust, C++** compile your source directly to machine code. The `.exe` *is* your
  program. Statically linked, it can be a single self-contained file with no runtime at all.
- **Python, JavaScript, Ruby** are interpreted. There is no machine code to produce. So
  the `.exe` is a **bootstrapper**: a small native launcher that carries the interpreter
  and your source (or bytecode) as an embedded payload, unpacks or memory-maps it, starts
  the interpreter, and hands it your entry script.

So "compiling Python to an exe" is a misnomer. Nothing is compiled. Your `.py` files are
turned into `.pyc` bytecode, zipped up with `python312.dll` and every package you import,
and a small C launcher is glued to the front. That is a PyInstaller build.

(The exception is **Nuitka**, which genuinely translates Python to C and compiles it.
Faster startup, real obfuscation, much slower and more fragile builds. Not what we need.)

### `.pyd` files — the thing that makes this harder than it sounds

A pure-Python package is just text files; bundling it is trivial. But packages like
`ctranslate2`, `onnxruntime` and `numpy` are C or C++ extensions. On Windows they ship as
`.pyd` files, which are **DLLs with a different extension**. Those `.pyd` files in turn
import *other* native DLLs — BLAS libraries, CUDA runtimes, MSVC redistributables.

So a Python bundler cannot just copy `.py` files. It has to open every `.pyd`, read its
import table, find those DLLs on your build machine, copy them in too, and repeat
transitively. This is where builds break, and it is the single biggest source of "works
on my machine, crashes on theirs."

---

## 3. The four families of desktop architecture

### A. Native UI toolkits

Win32/WinUI on Windows, Cocoa on macOS, GTK on Linux. Real OS widgets, tiny binaries,
best performance and accessibility. But you write a separate UI per platform and none of
your web skills transfer. Rarely the right call for a solo developer with an existing web UI.

### B. Cross-platform UI frameworks

**Qt** (C++/Python), **Flutter**, **Avalonia** (.NET), **.NET MAUI**, **Kotlin Multiplatform**.
One codebase, their own rendering engine, real app windows. Excellent results, but the UI
is a full rewrite in their widget model. Months of work if you already have HTML.

### C. Web-tech shells — you keep your HTML/CSS/JS

This is the on-ramp for web developers. Your existing frontend runs unchanged inside a
window that is really a browser engine.

| | **Electron** | **Tauri** | **pywebview** |
| --- | --- | --- | --- |
| Browser engine | Bundles Chromium | Uses OS webview (WebView2 / WKWebView / WebKitGTK) | Uses OS webview |
| Backend language | Node.js | Rust | **Python** |
| Base size | ~150 MB | ~5 MB | ~2 MB (plus Python) |
| Rendering consistency | Identical everywhere — you ship the engine | Varies by OS version | Varies by OS version |
| Used by | VS Code, Slack, Discord, Figma | newer indie apps | Python desktop tools |

Electron is the safest and heaviest. Tauri is the modern lightweight choice but its
backend is Rust — wrong language for us. **pywebview** is the interesting one here: it
gives a Python app a real native window rendering your HTML, so it would suit this project
if we later want an app window instead of a browser tab.

### D. Local server + the user's own browser

Start a normal HTTP server bound to `127.0.0.1`, open the user's default browser at it.
No UI framework at all. Jupyter, Stable Diffusion WebUI, Ollama's UI, and countless
internal tools ship exactly this way.

**Advantages:** zero new UI code, zero new dependencies, your existing dev workflow is
unchanged, and debugging is just DevTools as usual.

**Disadvantages:** it appears as a browser tab rather than an app window, so there is no
taskbar identity, no native menu, and quitting means closing a separate console window.
It also feels less "installed" to users who care about that.

---

## 4. Why this project uses (D), and what it costs

This repo already contains:

- a working HTTP server (`app/main.py`, FastAPI + uvicorn),
- a complete frontend (`app/static/index.html`, `app.js`, `styles.css`),
- all the actual work happening server-side in Python (`app/transcriber.py`).

Option D costs **no UI rewrite and no new dependency**. Options B and C would each mean
rebuilding or re-hosting a frontend that already works, to gain a window frame.

The honest trade-off: the user gets a browser tab plus a small console window that must
stay open. If that feels unpolished later, upgrading to **pywebview** is roughly 30 lines —
swap `webbrowser.open(url)` for `webview.create_window(...)` and the same server, same
HTML, gets a real native window. The packaging work described below does not change.

---

## 5. Python packaging tools

| Tool | Approach | Notes |
| --- | --- | --- |
| **PyInstaller** | Bundles interpreter + bytecode + native deps | Most widely used, best library support via its hook system. **Our choice.** |
| **Nuitka** | Compiles Python → C → machine code | Faster startup, real compilation, slow and finicky builds |
| **cx_Freeze** | Similar to PyInstaller | Smaller ecosystem, fewer library hooks |
| **py2exe** | Windows-only, older | Largely superseded |
| **Briefcase** (BeeWare) | Bundling + installers + app stores | Opinionated, aimed at GUI toolkit apps |
| **embeddable Python** | Ship the official ZIP distribution + your source in a folder | No build step, but a fragile folder the user can break |

PyInstaller wins because ctranslate2, onnxruntime, faster-whisper and uvicorn are all
awkward to bundle, and PyInstaller has the largest body of existing hooks for exactly
those awkward cases.

---

## 6. How PyInstaller works, in detail

Worth understanding properly, because when a build fails this is where you debug.

### Phase 1 — Analysis

PyInstaller imports your entry script and walks the **import graph statically**. It reads
your source, finds `import x`, opens `x`, finds its imports, and recurses.

Static analysis has an obvious blind spot: **anything imported dynamically is invisible**.

```python
import uvicorn.protocols.http.h11_impl          # found — a literal import
mod = importlib.import_module(f"uvicorn.protocols.http.{name}")   # NOT found
```

uvicorn, FastAPI and onnxruntime all select implementations dynamically at runtime. Those
modules are silently omitted, and your exe dies with `ModuleNotFoundError` — on the user's
machine, not yours. The fix is the `hiddenimports` list: modules you promise are needed
even though nothing visibly imports them.

### Phase 2 — Native dependency scan

Every `.pyd` PyInstaller collected gets its PE import table read, and the DLLs it names
are located on your build machine and copied in. Transitively. This step is why the build
must run on the target OS.

### Phase 3 — Hooks

PyInstaller ships hundreds of per-package **hook files** encoding hard-won knowledge
("numpy also needs these DLLs", "PIL needs these plugins"). You can write your own, and
the helpers are the useful part:

```python
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

datas   = collect_data_files("faster_whisper")     # the Silero VAD .onnx files
binaries = collect_dynamic_libs("onnxruntime")     # its native DLLs
hidden  = collect_submodules("uvicorn")            # every submodule, dynamic or not
```

### Phase 4 — Assembly

Bytecode goes into a **PYZ** (an encrypted-or-not zip). PYZ, binaries, and data files go
into either one exe or a folder, wrapped by the bootloader — a small C program that is
the only genuinely compiled part of your build.

### The `.spec` file

A `.spec` file **is a Python script**, executed by PyInstaller. It is not config:

```python
a = Analysis(
    ['app/launcher.py'],              # entry point
    binaries=[('ffmpeg.exe', '.')],   # (source_on_build_machine, dest_inside_bundle)
    datas=[('app/static', 'app/static')],
    hiddenimports=['uvicorn.protocols.http.h11_impl'],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, ...)
coll = COLLECT(exe, a.binaries, a.datas, name='speech2text')   # onedir only
```

Because it is Python, you can compute those lists — call `collect_data_files`, glob a
directory, branch on an environment variable. Generate it once with `pyi-makespec`, then
edit and commit it; the `.spec` is real source code, not a build artifact.

### onefile vs onedir — a decision with real consequences

**`--onefile`** produces a single `.exe` that is a self-extracting archive. Every launch
it unpacks its entire payload to a temp directory, runs, then deletes it.

- One file to hand over. Feels clean.
- **Every startup pays the extraction cost.** For this app (~500 MB of ctranslate2,
  onnxruntime and ffmpeg) that is 15–20 seconds *every time*, and a burst of disk writes
  that some antivirus products treat as suspicious.

**`--onedir`** produces a folder: your `.exe` plus an `_internal\` directory of DLLs and
data.

- Starts in about two seconds because nothing is unpacked.
- It is a folder, not a file — but you never hand a user a folder. You hand them an
  **installer** that lays it down for them.

**Rule of thumb:** onefile for a small CLI tool, onedir + installer for anything real.

### What breaks under freezing

```python
import sys

if getattr(sys, "frozen", False):     # True only inside a PyInstaller bundle
    base = sys._MEIPASS               # onefile: the temp dir; onedir: the _internal dir
else:
    base = os.path.dirname(__file__)
```

Concretely:

- **`__file__`** points inside the bundle, which is temporary (onefile) or read-only in
  spirit (onedir). Never write next to it.
- **`sys.executable`** is now *your exe*, not `python.exe`. Any code that re-launches the
  interpreter breaks.
- **`os.getcwd()`** is wherever the user's shortcut happened to point.
- **`multiprocessing`** must call `multiprocessing.freeze_support()` first thing in
  `__main__`, or each child process re-runs your whole program. (We use a thread, so this
  does not bite us — but it will the day you add a process pool.)
- **Relative paths in general** are meaningless. Resolve everything explicitly.

This project hits the `__file__` problem directly: `app/config.py` computes
`PROJECT_ROOT = Path(__file__).resolve().parent.parent` and puts `models/` and `work/`
under it. Frozen, that is a temp directory — the 1.6 GB model would be re-downloaded on
every launch and thrown away. The fix is in §10.

---

## 7. Where a Windows app is allowed to write

This trips up everyone coming from Linux servers, where your app owns its own directory.

| Path | Env var | Writable without admin? | Use for |
| --- | --- | --- | --- |
| `C:\Program Files\...` | `%ProgramFiles%` | **No** | Installed program files only |
| `C:\Users\x\AppData\Local\...` | `%LOCALAPPDATA%` | Yes | Caches, models, machine-specific data |
| `C:\Users\x\AppData\Roaming\...` | `%APPDATA%` | Yes | Small settings that follow the user across machines |
| `C:\Users\x\Documents` | — | Yes | Files the *user* considers theirs |
| `C:\Users\x\AppData\Local\Temp` | `%TEMP%` | Yes | Scratch, may vanish |

**Roaming vs Local** matters in corporate domains: `%APPDATA%` is synced to the network
when the user logs into another machine, `%LOCALAPPDATA%` is not. Never put a 1.6 GB model
in Roaming — put caches in Local.

For this app: model weights and scratch audio go in `%LOCALAPPDATA%\speech2text\`.

---

## 8. Installers — the step after packaging

Packaging produces a program. Distribution needs an **installer**, which copies files into
place, creates a Start Menu shortcut, registers an uninstaller so the app appears in
"Apps & features", and optionally associates file types.

| Tool | Output | Notes |
| --- | --- | --- |
| **Inno Setup** | one `Setup.exe` | Free, one readable script, by far the easiest. **Our choice.** |
| **NSIS** | one `Setup.exe` | Free, more scriptable, uglier syntax |
| **WiX Toolset** | `.msi` | The Microsoft-blessed route; enterprise deployment wants MSI; steep |
| **MSIX** | `.msix` | Modern, sandboxed, Store-compatible; needs signing to install at all |

### Per-user vs per-machine

- **Per-machine** installs to `Program Files`, needs admin, triggers a **UAC** prompt
  (the "Do you want to allow this app to make changes?" dialog), available to all users.
- **Per-user** installs to `%LOCALAPPDATA%\Programs\`, **needs no admin and shows no UAC
  prompt**, available to that user only.

For a non-technical single user, per-user is strictly better: one less scary dialog, and
it works even on a locked-down work laptop. In Inno Setup that is `PrivilegesRequired=lowest`.

---

## 9. Code signing and SmartScreen — the part nobody warns you about

Hand someone an unsigned `.exe` downloaded from the internet and Windows shows a blue
full-screen box:

> **Windows protected your PC**
> Microsoft Defender SmartScreen prevented an unrecognized app from starting.

There is no visible "Run" button. The user must click **More info**, then **Run anyway**.
Non-technical users read this as "this is a virus" and stop. Plan for it.

**Why it happens:** SmartScreen scores files by reputation, keyed on the signing
certificate (or the file hash, if unsigned). A brand-new unsigned binary has no reputation.

**Options, cheapest first:**

1. **Do nothing, warn them.** Fine when you are handing the file to one person you can
   talk to. Tell them in advance: "Windows will warn you — click More info, then Run anyway."
   Give it to them on a USB stick or a direct transfer rather than a download, which also
   avoids the separate "mark of the web" block.
2. **Azure Trusted Signing** — roughly $10/month, no hardware token, integrates with CI.
   Currently the cheapest real path for an individual.
3. **OV certificate** — ~$200–400/year from a CA. Since 2023 the private key must live on
   a hardware token or HSM, which complicates automated builds. Removes the warning only
   after the certificate accumulates reputation across some installs.
4. **EV certificate** — ~$400–700/year, hardware token mandatory, but grants SmartScreen
   reputation **immediately**. What commercial software uses.

**Antivirus false positives** are a related nuisance. PyInstaller's bootloader unpacking
an archive and executing code from temp looks, structurally, like malware, and the
bootloader is reused by actual malware. Two mitigations: **do not use UPX compression**
(`upx=False`), which markedly increases detection rates, and prefer **onedir** over
onefile. If a specific product still flags you, submit a false-positive report to that vendor.

---

## 10. This project, concretely

### What happens when the user double-clicks

```mermaid
flowchart TD
    A[User double-clicks the Start Menu shortcut] --> B[speech2text.exe bootloader starts]
    B --> C[Bundled python312.dll boots, runs app/launcher.py]
    C --> D[Create %LOCALAPPDATA%/speech2text/{models,work,logs}]
    D --> E[Set S2T_MODEL_DIR and S2T_WORK_DIR to those paths]
    E --> F[Prepend bundled binary dir to PATH so ffmpeg/ffprobe resolve]
    F --> G{nvcuda.dll loadable?}
    G -->|yes, DLLs not cached| H[Download cuDNN + cuBLAS wheels from PyPI, verify SHA-256, extract DLLs]
    G -->|no NVIDIA driver| I[Skip - CPU path]
    H --> J[os.add_dll_directory on the CUDA folder]
    I --> K[Bind 127.0.0.1:0 to claim a free port]
    J --> K
    K --> L[Start uvicorn on a background thread]
    L --> M[webbrowser.open on the localhost URL]
    M --> N[Console window stays open: 'close this window to stop']
    N --> O[User drops in an MP3; transcriber picks CUDA or CPU automatically]
```

### The four problems the launcher solves

**1. Writable paths.** As shown in §6, `config.PROJECT_ROOT` is derived from `__file__`
and is useless when frozen. The launcher sets `S2T_MODEL_DIR` and `S2T_WORK_DIR`
environment variables **before** anything imports `app.config`. That ordering is
load-bearing: `get_settings()` is decorated with `@lru_cache`, so the first call freezes
the configuration for the life of the process. No change to `config.py` is needed —
the existing `os.environ.get(...)` reads pick it up.

**2. ffmpeg.** `app/audio.py` calls `shutil.which("ffmpeg")` and runs bare `"ffprobe"`.
Both consult `PATH`. So bundling `ffmpeg.exe` and `ffprobe.exe` and prepending their
folder to `os.environ["PATH"]` makes both work with **zero changes to `audio.py`**.

This works because ffmpeg is launched as a **separate process**, and process launch still
searches `PATH`. Which leads directly to the next point.

**3. CUDA DLLs — and why `PATH` is the wrong tool here.**

Since Python 3.8, extension modules **no longer use `PATH`** to find their dependent DLLs;
that was closed as a security hole (`PATH` is user-writable, so it was a hijacking vector).
The replacement is explicit:

```python
os.add_dll_directory(r"C:\Users\x\AppData\Local\speech2text\cuda")
```

So: `PATH` for ffmpeg (a subprocess), `add_dll_directory` for cuDNN (a DLL loaded by a
`.pyd`). Same goal, two different mechanisms, and using the wrong one fails silently.

The CUDA runtime is genuinely large — measured in this repo's venv, `cudnn` is 976 MB and
`cublas` 528 MB installed, with `libcudnn_engines_precompiled` alone at 570 MB. Bundling
it would make a ~2 GB installer, most of it dead weight for a user with no NVIDIA card.
Instead the launcher probes for `nvcuda.dll` — a DLL installed by the NVIDIA graphics
driver, present only on machines with an NVIDIA GPU — and downloads the DLLs from PyPI on
first run only when one is found. No card, no network, or a hash mismatch: it logs and
returns, and the app runs on CPU.

**4. The GPU/CPU decision itself is already written.** `app/transcriber.py:114`:

```python
def resolve_device(self) -> str:
    if self.settings.device != "auto":
        return self.settings.device
    return "cuda" if cuda_device_count() > 0 else "cpu"
```

and lines 146–168 already catch a failed CUDA model load and retry on `cpu`/`int8`. So
"fast GPU by default, CPU if there is no GPU" needs no new logic — only the DLLs in place.

### Build pipeline

PyInstaller **cannot cross-compile**. It bundles the interpreter and native DLLs *from the
machine it runs on*, so a Windows exe must be built on Windows. You are on WSL2, which
means the Windows side of the same box is the build machine. The alternative is a free
GitHub Actions `windows-latest` runner.

```powershell
# On Windows (not WSL) — once
winget install Python.Python.3.12
winget install JRSoftware.InnoSetup

cd C:\path\to\speech2text
py -3.12 -m venv .venv-win
.venv-win\Scripts\pip install -r requirements.txt pyinstaller
# note: NOT requirements-gpu.txt — CUDA is fetched at runtime instead

# Fetch a static ffmpeg build and drop ffmpeg.exe + ffprobe.exe into packaging\bin\

# Build the program
.venv-win\Scripts\pyinstaller packaging\speech2text.spec --noconfirm
# -> dist\speech2text\speech2text.exe  plus  dist\speech2text\_internal\

# Wrap it in an installer
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
# -> dist\speech2text-setup.exe   <- the single file you send
```

### What gets tested where

The Python unit tests stay on Linux and keep running under `pytest` as they do now — path
resolution, GPU detection with a mocked `WinDLL`, wheel extraction against a fixture zip,
free-port selection. What **cannot** be unit-tested is the bundle itself: whether a hidden
import was missed, whether a DLL is absent, whether the model downloads. That needs one
manual run of the built exe on Windows, ideally on a machine that is not your dev box.

---

## 11. Gotchas checklist

Things that will bite you, roughly in the order they will:

- [ ] `ModuleNotFoundError` at runtime but not in dev → dynamic import, add to `hiddenimports`
- [ ] Missing data file → packages ship non-`.py` assets (`faster_whisper`'s Silero `.onnx` VAD models); use `collect_data_files`
- [ ] `__file__`-relative writes → temp directory, silently lost; redirect to `%LOCALAPPDATA%`
- [ ] DLL not found → `os.add_dll_directory`, not `PATH`
- [ ] Subprocess not found → `PATH`, not `add_dll_directory`
- [ ] Slow startup → you used onefile; switch to onedir + installer
- [ ] Antivirus flags the build → turn UPX off, prefer onedir, report false positives
- [ ] SmartScreen blocks the user → expected when unsigned; warn them, or sign it
- [ ] Works on your machine only → you built with a system dependency installed that they lack; test on a clean VM
- [ ] `multiprocessing` spawns infinite copies of your app → `freeze_support()`
- [ ] Installer needs admin → set per-user install to skip UAC
- [ ] Console window confuses the user → it is also their only quit button; label it clearly, or move to a tray icon
- [ ] Huge bundle → you installed GPU/dev extras into the build venv; keep the build venv minimal

---

## 12. Glossary

**PE** — Portable Executable, the Windows binary format for `.exe`, `.dll`, `.pyd`.
**DLL** — a dynamically loaded library. `.so` on Linux, `.dylib` on macOS.
**`.pyd`** — a Windows DLL that exposes a Python C extension module.
**Static linking** — dependencies compiled into the binary. **Dynamic linking** — resolved at load time.
**Bootloader** — the small native launcher PyInstaller prepends; the only compiled code in a Python bundle.
**Frozen** — a Python app bundled with its interpreter. `sys.frozen` is `True` inside one.
**`sys._MEIPASS`** — where a frozen app's bundled files live: a temp dir (onefile) or `_internal\` (onedir).
**Hidden import** — a module used at runtime that static analysis cannot see.
**Hook** — a PyInstaller plugin describing how to bundle a specific library.
**UAC** — User Account Control, the Windows admin-elevation prompt.
**SmartScreen** — the Windows reputation check that blocks unrecognised executables.
**Code signing** — attaching a cryptographic signature so the OS can attribute the binary to you.
**MSI/MSIX** — Microsoft's structured installer formats, as opposed to a self-contained `Setup.exe`.
**WebView2** — the Chromium-based embeddable browser control shipped with Windows; what Tauri and pywebview use there.
