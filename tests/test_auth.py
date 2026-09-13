"""The password gate: what it blocks, what it lets through, and the lockout.

The Gatekeeper is exercised directly where clock control matters, and through
the app everywhere else — a rule that only holds in the unit test would not
protect the actual endpoints.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.auth import SESSION_COOKIE, Gatekeeper
from app.config import get_settings, load_dotenv

PASSWORD = "test1234"


@pytest.fixture
def locked_client(make_client):
    """A client for an app that demands a password."""
    with make_client(S2T_PASSWORD=PASSWORD) as client:
        yield client


def login(client, password: str = PASSWORD):
    return client.post("/api/login", data={"password": password})


# -- the gate itself ------------------------------------------------------


def test_no_password_means_no_gate(client) -> None:
    """The default install stays open — the gate is opt-in."""
    assert client.get("/").status_code == 200
    assert client.get("/api/models").status_code == 200
    # The form is pointless without a password, so it bounces back to the app.
    assert client.get("/login", follow_redirects=False).status_code == 303


def test_pages_redirect_to_the_login_form(locked_client) -> None:
    response = locked_client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"
    assert locked_client.get("/login").status_code == 200


def test_the_api_is_closed_without_a_session(locked_client) -> None:
    for path in ("/api/models", "/api/health", "/api/jobs/whatever"):
        assert locked_client.get(path).status_code == 401, path


def test_uploads_are_closed_without_a_session(locked_client, tone_wav: Path) -> None:
    with tone_wav.open("rb") as handle:
        response = locked_client.post(
            "/api/transcribe",
            files={"file": (tone_wav.name, handle, "audio/wav")},
            data={"model": "small", "language": "ar", "task": "transcribe"},
        )
    assert response.status_code == 401


def test_the_login_page_and_its_assets_stay_reachable(locked_client) -> None:
    """Otherwise the form would render unstyled and unscripted."""
    for path in ("/login", "/static/styles.css", "/static/login.js", "/static/logo.svg"):
        assert locked_client.get(path).status_code == 200, path


def test_the_right_password_opens_everything(locked_client) -> None:
    assert login(locked_client).status_code == 200
    assert locked_client.cookies.get(SESSION_COOKIE)
    assert locked_client.get("/").status_code == 200
    assert locked_client.get("/api/models").status_code == 200


def test_the_wrong_password_is_refused(locked_client) -> None:
    response = login(locked_client, "wrong")
    assert response.status_code == 401
    assert response.json()["remaining"] == 2
    assert not locked_client.cookies.get(SESSION_COOKIE)


def test_the_countdown_reads_as_arabic_not_as_a_number(locked_client) -> None:
    """Arabic has a dual form: "بقيت 2 محاولة" is not a sentence."""
    assert "محاولتان" in login(locked_client, "wrong").json()["detail"]
    assert "محاولة واحدة" in login(locked_client, "wrong").json()["detail"]
    assert "دقيقة" in login(locked_client, "wrong").json()["detail"]


def test_logging_in_again_after_a_slip_still_works(locked_client) -> None:
    """Failures reset on success, so a typo cannot creep toward a lockout."""
    login(locked_client, "wrong")
    assert login(locked_client).status_code == 200
    assert login(locked_client, "wrong").json()["remaining"] == 2


def test_logout_ends_the_session(locked_client) -> None:
    login(locked_client)
    assert locked_client.post("/api/logout").status_code == 204
    assert locked_client.get("/", follow_redirects=False).status_code == 303


def test_a_forged_cookie_is_rejected(locked_client) -> None:
    forged = f"{int(time.time()) + 3600}.{'0' * 64}"
    locked_client.cookies.set(SESSION_COOKIE, forged)
    assert locked_client.get("/api/models").status_code == 401


# -- the lockout ----------------------------------------------------------


def test_three_wrong_guesses_lock_the_door(locked_client) -> None:
    assert login(locked_client, "wrong").status_code == 401
    assert login(locked_client, "wrong").status_code == 401

    third = login(locked_client, "wrong")
    assert third.status_code == 429
    assert third.json()["locked"] is True
    assert int(third.headers["Retry-After"]) > 0


def test_the_lock_ignores_the_correct_password(locked_client) -> None:
    """A lockout that the right password opens is not a lockout."""
    for _ in range(3):
        login(locked_client, "wrong")

    response = login(locked_client)
    assert response.status_code == 429
    assert not locked_client.cookies.get(SESSION_COOKIE)
    assert locked_client.get("/api/models").status_code == 401


def test_the_lock_lifts_when_the_window_passes() -> None:
    gate = Gatekeeper(password=PASSWORD, max_attempts=3, lockout_seconds=60)
    now = 1_000.0

    for _ in range(3):
        gate.attempt("10.0.0.1", "wrong", now=now)
    assert gate.attempt("10.0.0.1", PASSWORD, now=now).locked

    later = now + 61
    result = gate.attempt("10.0.0.1", PASSWORD, now=later)
    assert result.ok and result.token


def test_one_client_cannot_lock_out_another() -> None:
    gate = Gatekeeper(password=PASSWORD, max_attempts=3)
    for _ in range(3):
        gate.attempt("10.0.0.1", "wrong")

    assert gate.attempt("10.0.0.1", PASSWORD).locked
    assert gate.attempt("10.0.0.2", PASSWORD).ok


def test_expired_sessions_stop_working() -> None:
    gate = Gatekeeper(password=PASSWORD, session_seconds=60)
    token = gate.issue(now=1_000.0)
    assert gate.accepts(token, now=1_050.0)
    assert not gate.accepts(token, now=1_100.0)


@pytest.mark.parametrize("token", ["", None, "nonsense", "abc.def", "9999999999."])
def test_malformed_tokens_are_rejected(token) -> None:
    gate = Gatekeeper(password=PASSWORD)
    assert not gate.accepts(token)


# -- .env loading ---------------------------------------------------------


def test_the_env_file_supplies_the_password(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "\n"
        "export S2T_PASSWORD='from-file'\n"
        'S2T_LOCKOUT_SECONDS="30"\n'
        "not a pair\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("S2T_ENV_FILE", str(env_file))
    # An earlier load_dotenv() may have copied the developer's own .env into the
    # process; clear the keys under test so the file is what decides.
    for key in ("S2T_PASSWORD", "S2T_LOCKOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.password == "from-file"
    assert settings.lockout_seconds == 30
    assert settings.auth_enabled
    get_settings.cache_clear()


def test_the_environment_wins_over_the_env_file(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("S2T_PASSWORD=from-file\n", encoding="utf-8")
    monkeypatch.setenv("S2T_PASSWORD", "from-env")

    load_dotenv(env_file)

    import os

    assert os.environ["S2T_PASSWORD"] == "from-env"


def test_a_missing_env_file_is_not_an_error(tmp_path: Path) -> None:
    load_dotenv(tmp_path / "nope.env")
