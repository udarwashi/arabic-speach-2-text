"""HTTP surface: upload, stream, download.

``create_app`` takes an optional transcriber so tests can inject a fake and run
without downloading a model.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from . import __version__
from .audio import SUPPORTED_EXTENSIONS, AudioError, is_supported_extension, require_ffmpeg
from .auth import SESSION_COOKIE, Gatekeeper
from .config import (
    DEFAULT_MODEL,
    LANGUAGES,
    MODEL_REGISTRY,
    TASKS,
    Settings,
    get_settings,
)
from .formats import FORMATTERS, render
from .jobs import Job, JobRegistry
from .transcriber import SupportsTranscription, Transcriber
from .worker import run_job

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_CHUNK = 1024 * 1024
PURGE_INTERVAL_SECONDS = 60

# Reachable without a session, because the login page itself needs them.
PUBLIC_PATHS = frozenset({"/login", "/api/login", "/favicon.ico"})


def _arabic_attempts(count: int) -> str:
    """Arabic counts one, two and many differently; "بقيت 2 محاولة" reads wrong."""
    if count == 1:
        return "بقيت محاولة واحدة"
    if count == 2:
        return "بقيت محاولتان"
    return f"بقيت {count} محاولات"


def _arabic_minutes(seconds: int) -> str:
    minutes = max(1, round(seconds / 60))
    if minutes == 1:
        return "دقيقة"
    if minutes == 2:
        return "دقيقتين"
    if minutes <= 10:
        return f"{minutes} دقائق"
    return f"{minutes} دقيقة"


def create_app(transcriber: SupportsTranscription | None = None) -> FastAPI:
    settings = get_settings()
    gate = Gatekeeper(
        password=settings.password,
        max_attempts=settings.max_attempts,
        lockout_seconds=settings.lockout_seconds,
        session_seconds=settings.session_seconds,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.gate = gate
        app.state.transcriber = transcriber or Transcriber(settings)
        app.state.registry = JobRegistry(ttl_seconds=settings.job_ttl_seconds)
        # One decode at a time: a 4 GB GPU cannot hold two large models.
        app.state.semaphore = asyncio.Semaphore(1)
        app.state.purge_task = asyncio.create_task(_purge_loop(app.state.registry))
        try:
            require_ffmpeg()
        except AudioError as exc:
            logger.warning("ffmpeg check failed at startup: %s", exc.detail)
        try:
            yield
        finally:
            app.state.purge_task.cancel()
            running = [
                job.task_handle
                for job in app.state.registry.all()
                if job.task_handle is not None and not job.task_handle.done()
            ]
            for handle in running:
                handle.cancel()
            # Await the cancellations so each worker's cleanup runs and no scratch
            # file is left behind on shutdown.
            if running:
                await asyncio.gather(*running, return_exceptions=True)

    app = FastAPI(
        title="تحويل الصوت إلى نص",
        description="Arabic speech-to-text powered by faster-whisper.",
        version=__version__,
        lifespan=lifespan,
    )

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # -- password gate -----------------------------------------------------

    @app.middleware("http")
    async def require_session(request: Request, call_next):
        """Nothing but the login page and its assets is reachable unauthenticated."""
        path = request.url.path
        if not gate.enabled or path in PUBLIC_PATHS or path.startswith("/static/"):
            return await call_next(request)
        if gate.accepts(request.cookies.get(SESSION_COOKIE)):
            return await call_next(request)
        if path.startswith("/api/"):
            return JSONResponse({"detail": "الجلسة منتهية. سجّل الدخول من جديد."}, 401)
        # A browser asking for a page gets sent to the form, not a bare 401.
        return RedirectResponse("/login", status_code=303)

    @app.get("/login", include_in_schema=False)
    async def login_page(request: Request) -> Response:
        if not gate.enabled or gate.accepts(request.cookies.get(SESSION_COOKIE)):
            return RedirectResponse("/", status_code=303)
        page = STATIC_DIR / "login.html"
        if not page.is_file():  # pragma: no cover - only if the build is broken
            raise HTTPException(status_code=500, detail="صفحة الدخول غير موجودة.")
        return FileResponse(page, media_type="text/html; charset=utf-8")

    @app.post("/api/login", include_in_schema=False)
    async def login(request: Request, password: str = Form("")) -> JSONResponse:
        if not gate.enabled:
            return JSONResponse({"ok": True})

        client = request.client.host if request.client else "unknown"
        result = gate.attempt(client, password)

        if result.locked:
            response = JSONResponse(
                {
                    "ok": False,
                    "locked": True,
                    "retry_after": result.locked_seconds,
                    "detail": (
                        f"تم قفل الدخول بعد {settings.max_attempts} محاولات خاطئة. "
                        f"حاول مرة أخرى بعد {_arabic_minutes(result.locked_seconds)}."
                    ),
                },
                status_code=429,
            )
            response.headers["Retry-After"] = str(result.locked_seconds)
            return response

        if not result.ok:
            return JSONResponse(
                {
                    "ok": False,
                    "locked": False,
                    "remaining": result.remaining,
                    "detail": (
                        "كلمة المرور غير صحيحة. "
                        f"{_arabic_attempts(result.remaining)} قبل القفل."
                    ),
                },
                status_code=401,
            )

        response = JSONResponse({"ok": True})
        response.set_cookie(
            SESSION_COOKIE,
            result.token,
            max_age=settings.session_seconds,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return response

    @app.post("/api/logout", include_in_schema=False)
    async def logout() -> Response:
        response = Response(status_code=204)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    # -- dependencies ------------------------------------------------------

    def get_registry(request: Request) -> JobRegistry:
        return request.app.state.registry

    def get_transcriber(request: Request) -> SupportsTranscription:
        return request.app.state.transcriber

    def get_app_settings(request: Request) -> Settings:
        return request.app.state.settings

    def require_job(job_id: str, registry: JobRegistry = Depends(get_registry)) -> Job:
        job = registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="لا توجد مهمة بهذا المعرّف.")
        return job

    # -- pages -------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        page = STATIC_DIR / "index.html"
        if not page.is_file():  # pragma: no cover - only if the build is broken
            raise HTTPException(status_code=500, detail="واجهة المستخدم غير موجودة.")
        return FileResponse(page, media_type="text/html; charset=utf-8")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    # -- metadata ----------------------------------------------------------

    @app.get("/api/health")
    async def health(
        request: Request,
        registry: JobRegistry = Depends(get_registry),
        transcriber: SupportsTranscription = Depends(get_transcriber),
        app_settings: Settings = Depends(get_app_settings),
    ) -> JSONResponse:
        jobs = registry.all()
        return JSONResponse(
            {
                "status": "ok",
                "version": __version__,
                "runtime": transcriber.runtime_info(),
                "auth": gate.enabled,
                "settings": {
                    "default_model": app_settings.default_model,
                    "beam_size": app_settings.beam_size,
                    "max_upload_mb": app_settings.max_upload_mb,
                    "job_ttl_seconds": app_settings.job_ttl_seconds,
                },
                "jobs": {
                    "total": len(jobs),
                    "active": sum(1 for job in jobs if not job.status.is_terminal),
                },
            }
        )

    @app.get("/api/models")
    async def models(app_settings: Settings = Depends(get_app_settings)) -> JSONResponse:
        return JSONResponse(
            {
                "default_model": app_settings.default_model or DEFAULT_MODEL,
                "models": [
                    {
                        "key": spec.key,
                        "label": spec.label,
                        "params": spec.params,
                        "download": spec.download,
                        "note": spec.note,
                    }
                    for spec in MODEL_REGISTRY.values()
                ],
                "tasks": [{"key": key, "label": label} for key, label in TASKS.items()],
                "languages": [
                    {"key": key, "label": label} for key, label in LANGUAGES.items()
                ],
                "extensions": sorted(SUPPORTED_EXTENSIONS),
                "max_upload_mb": app_settings.max_upload_mb,
            }
        )

    # -- transcription -----------------------------------------------------

    @app.post("/api/transcribe", status_code=202)
    async def transcribe(
        file: UploadFile,
        model: str = Form(DEFAULT_MODEL),
        language: str = Form("ar"),
        task: str = Form("transcribe"),
        registry: JobRegistry = Depends(get_registry),
        transcriber: SupportsTranscription = Depends(get_transcriber),
        app_settings: Settings = Depends(get_app_settings),
    ) -> JSONResponse:
        if model not in MODEL_REGISTRY:
            raise HTTPException(status_code=400, detail=f"نموذج غير معروف: {model}")
        if language not in LANGUAGES:
            raise HTTPException(status_code=400, detail=f"لغة غير مدعومة: {language}")
        if task not in TASKS:
            raise HTTPException(status_code=400, detail=f"عملية غير مدعومة: {task}")

        filename = Path(file.filename or "audio").name
        if not is_supported_extension(filename):
            raise HTTPException(
                status_code=415,
                detail=(
                    "صيغة الملف غير مدعومة. الصيغ المدعومة: "
                    + "، ".join(sorted(SUPPORTED_EXTENSIONS))
                ),
            )

        job = registry.create(
            filename=filename, model=model, language=language, task=task
        )
        source = app_settings.work_dir / f"{job.id}{Path(filename).suffix.lower()}"
        try:
            await _save_upload(file, source, app_settings.max_upload_bytes)
        except HTTPException:
            registry.drop(job.id)
            raise

        job.task_handle = asyncio.create_task(
            run_job(
                registry=registry,
                job=job,
                transcriber=transcriber,
                source_path=source,
                semaphore=app.state.semaphore,
            )
        )
        return JSONResponse(job.as_dict(include_segments=False), status_code=202)

    @app.get("/api/jobs/{job_id}")
    async def job_state(job: Job = Depends(require_job)) -> JSONResponse:
        return JSONResponse(job.as_dict())

    @app.get("/api/jobs/{job_id}/stream")
    async def job_stream(
        job_id: str, registry: JobRegistry = Depends(get_registry)
    ) -> StreamingResponse:
        if registry.get(job_id) is None:
            raise HTTPException(status_code=404, detail="لا توجد مهمة بهذا المعرّف.")

        async def event_source():
            try:
                async for message in registry.subscribe(job_id):
                    yield _format_sse(message["event"], message["data"])
            except asyncio.CancelledError:  # client went away
                raise
            except KeyError:  # job purged mid-stream
                yield _format_sse("error", {"error": "انتهت صلاحية المهمة."})

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # do not let a reverse proxy buffer SSE
            },
        )

    @app.get("/api/jobs/{job_id}/download")
    async def download(fmt: str = "txt", job: Job = Depends(require_job)) -> Response:
        fmt = fmt.lower()
        if fmt not in FORMATTERS:
            raise HTTPException(
                status_code=400,
                detail="الصيغة المطلوبة غير مدعومة: " + ", ".join(FORMATTERS),
            )
        if not job.segments:
            raise HTTPException(status_code=409, detail="لا يوجد نص بعد لهذه المهمة.")

        body, content_type = render(fmt, job.segments)
        return Response(
            content=body,
            media_type=content_type,
            headers={"Content-Disposition": _attachment(job, fmt)},
        )

    @app.delete("/api/jobs/{job_id}", status_code=200)
    async def cancel(
        job: Job = Depends(require_job), registry: JobRegistry = Depends(get_registry)
    ) -> JSONResponse:
        previous = job.status.value
        handle = job.task_handle
        if handle is not None and not handle.done():
            handle.cancel()
        elif not job.status.is_terminal:
            registry.cancel(job)
        registry.drop(job.id)
        # The job is gone either way; report what it was rather than claiming it
        # was cancelled when it had already finished.
        return JSONResponse(
            {"job_id": job.id, "status": "deleted", "previous_status": previous}
        )

    return app


# -- helpers ---------------------------------------------------------------


async def _save_upload(file: UploadFile, destination: Path, limit: int) -> int:
    """Stream the upload to disk, enforcing the size cap as we go.

    The cap is checked against bytes actually written, not ``Content-Length``,
    which a client controls.
    """
    written = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("wb") as handle:
            while chunk := await file.read(UPLOAD_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"حجم الملف يتجاوز الحد المسموح ({limit // (1024 * 1024)} ميجابايت).",
                    )
                handle.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="الملف المرسل فارغ.")
    return written


def _format_sse(event: str, data: dict) -> str:
    if event == "heartbeat":
        return ": keep-alive\n\n"
    # ensure_ascii=False keeps Arabic readable on the wire; the stream is UTF-8.
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _attachment(job: Job, fmt: str) -> str:
    """Content-Disposition that survives an Arabic source filename."""
    stem = Path(job.filename).stem or "transcript"
    ascii_fallback = f"transcript-{job.id[:8]}.{fmt}"
    encoded = quote(f"{stem}.{fmt}", safe="")
    return f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{encoded}"


async def _purge_loop(registry: JobRegistry) -> None:
    while True:
        await asyncio.sleep(PURGE_INTERVAL_SECONDS)
        removed = registry.purge_expired()
        if removed:
            logger.info("purged %d expired job(s)", removed)


app = create_app()
