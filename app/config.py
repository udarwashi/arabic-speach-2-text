"""Settings and the registry of Whisper models offered to the user.

Everything configurable lives here and is read once from ``S2T_*`` environment
variables (optionally by way of a ``.env`` file), so no other module needs to
touch ``os.environ``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelSpec:
    """One selectable Whisper checkpoint."""

    key: str
    label: str          # shown in the UI, in Arabic
    params: str
    download: str
    note: str
    # Compute type used on CUDA. large-v3 is quantized so it fits a 4 GB card.
    cuda_compute_type: str = "float16"


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "large-v3-turbo": ModelSpec(
        key="large-v3-turbo",
        label="سريع ودقيق (موصى به)",
        params="809M",
        download="~1.6 GB",
        note="أفضل توازن بين السرعة والدقة.",
        cuda_compute_type="float16",
    ),
    "large-v3": ModelSpec(
        key="large-v3",
        label="أبطأ وأدق",
        params="1550M",
        download="~3.1 GB",
        note="الأدق للعربية الصعبة واللهجات، ويحتاج وقتاً أطول.",
        # int8 weights with float16 compute: ~1.6 GB of VRAM instead of ~3.1 GB.
        cuda_compute_type="int8_float16",
    ),
    "small": ModelSpec(
        key="small",
        label="خفيف — يعمل على المعالج",
        params="244M",
        download="~0.5 GB",
        note="للأجهزة المحدودة أو التجربة السريعة؛ دقته أقل.",
        cuda_compute_type="float16",
    ),
}

DEFAULT_MODEL = "large-v3-turbo"

# Whisper task names, mapped to Arabic labels for the UI.
TASKS: dict[str, str] = {
    "transcribe": "تحويل الصوت إلى نص",
    "translate": "ترجمة إلى الإنجليزية",
}

# "ar" keeps Whisper from guessing; "auto" lets it detect.
LANGUAGES: dict[str, str] = {
    "ar": "العربية (مثبّتة)",
    "auto": "تحديد تلقائي",
}


def load_dotenv(path: Path | None = None) -> None:
    """Copy ``KEY=value`` lines from a ``.env`` file into the environment.

    A real environment variable always wins, so the file is a default and never
    an override — that keeps ``S2T_PASSWORD= ./run.sh`` and the test suite able
    to switch settings off. Deliberately tiny: no dependency, no interpolation.
    """
    env_file = path or Path(os.environ.get("S2T_ENV_FILE", PROJECT_ROOT / ".env"))
    try:
        raw = env_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration."""

    default_model: str
    device: str            # "auto" | "cuda" | "cpu"
    compute_type: str      # "auto" | any CTranslate2 compute type
    beam_size: int
    cpu_threads: int
    max_upload_bytes: int
    job_ttl_seconds: int
    work_dir: Path
    model_dir: Path | None
    password: str = ""       # empty = the app is open, no login page
    max_attempts: int = 3    # wrong guesses allowed before the lockout
    lockout_seconds: int = 900
    session_seconds: int = 12 * 3600

    @property
    def max_upload_mb(self) -> int:
        return self.max_upload_bytes // (1024 * 1024)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.password)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()

    default_model = os.environ.get("S2T_MODEL", DEFAULT_MODEL)
    if default_model not in MODEL_REGISTRY:
        raise ValueError(
            f"S2T_MODEL={default_model!r} is not one of {sorted(MODEL_REGISTRY)}"
        )

    work_dir = Path(os.environ.get("S2T_WORK_DIR", PROJECT_ROOT / "work"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # Default to a project-local directory so the weights are downloaded once and
    # always found again. Set S2T_MODEL_DIR="" to use the shared Hugging Face cache.
    raw_model_dir = os.environ.get("S2T_MODEL_DIR", str(PROJECT_ROOT / "models"))
    model_dir = Path(raw_model_dir) if raw_model_dir.strip() else None
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        default_model=default_model,
        device=os.environ.get("S2T_DEVICE", "auto"),
        compute_type=os.environ.get("S2T_COMPUTE_TYPE", "auto"),
        beam_size=_env_int("S2T_BEAM_SIZE", 5),
        cpu_threads=_env_int("S2T_CPU_THREADS", 0),  # 0 = let CTranslate2 decide
        max_upload_bytes=_env_int("S2T_MAX_UPLOAD_MB", 500) * 1024 * 1024,
        job_ttl_seconds=_env_int("S2T_JOB_TTL_SECONDS", 3600),
        work_dir=work_dir,
        model_dir=model_dir,
        password=os.environ.get("S2T_PASSWORD", "").strip(),
        max_attempts=_env_int("S2T_MAX_ATTEMPTS", 3),
        lockout_seconds=_env_int("S2T_LOCKOUT_SECONDS", 900),
        session_seconds=_env_int("S2T_SESSION_SECONDS", 12 * 3600),
    )
