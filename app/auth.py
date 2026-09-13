"""Single-password gate for the whole app.

There are no accounts: one shared password, taken from ``S2T_PASSWORD`` (which
normally lives in ``.env``), is exchanged for a signed session cookie. The
password itself is never stored in the cookie and never leaves the server.

Three wrong guesses in a row lock the guessing client out for a cooldown window.
Both the failure counters and the signing key live in memory, so restarting the
server clears every lock and invalidates every session — that restart is also the
documented way out if you lock yourself out.
"""

from __future__ import annotations

import hmac
import secrets
import time
from dataclasses import dataclass, field
from hashlib import sha256
from math import ceil

SESSION_COOKIE = "s2t_session"


def _sign(secret: bytes, message: str) -> str:
    return hmac.new(secret, message.encode("utf-8"), sha256).hexdigest()


@dataclass(frozen=True)
class Attempt:
    """The result of one login attempt."""

    ok: bool
    token: str | None = None
    # How many guesses are left before the lock closes.
    remaining: int = 0
    # Seconds until the lock lifts; 0 when not locked.
    locked_seconds: int = 0

    @property
    def locked(self) -> bool:
        return self.locked_seconds > 0


@dataclass
class Gatekeeper:
    """Checks passwords, issues sessions, and counts failures per client."""

    password: str
    max_attempts: int = 3
    lockout_seconds: int = 900
    session_seconds: int = 12 * 3600
    # A fresh key per process: sessions do not survive a restart, which is what
    # we want for a local tool with no session store.
    secret: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    _failures: dict[str, int] = field(default_factory=dict, repr=False)
    _locked_until: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def enabled(self) -> bool:
        """No password configured means no gate at all."""
        return bool(self.password)

    # -- locking -----------------------------------------------------------

    def locked_seconds(self, client: str, *, now: float | None = None) -> int:
        now = time.time() if now is None else now
        until = self._locked_until.get(client)
        if until is None:
            return 0
        if until <= now:
            # The window has passed: forget it, and give the client a clean slate.
            self._locked_until.pop(client, None)
            self._failures.pop(client, None)
            return 0
        return ceil(until - now)

    # -- attempts ----------------------------------------------------------

    def attempt(self, client: str, password: str, *, now: float | None = None) -> Attempt:
        now = time.time() if now is None else now

        locked = self.locked_seconds(client, now=now)
        if locked:
            return Attempt(ok=False, locked_seconds=locked)

        # compare_digest keeps the check from leaking the password's length or
        # its matching prefix through timing.
        if hmac.compare_digest(password, self.password):
            self._failures.pop(client, None)
            return Attempt(ok=True, token=self.issue(now=now))

        failures = self._failures.get(client, 0) + 1
        self._failures[client] = failures
        if failures >= self.max_attempts:
            self._locked_until[client] = now + self.lockout_seconds
            return Attempt(ok=False, locked_seconds=self.locked_seconds(client, now=now))
        return Attempt(ok=False, remaining=self.max_attempts - failures)

    # -- sessions ----------------------------------------------------------

    def issue(self, *, now: float | None = None) -> str:
        now = time.time() if now is None else now
        expires = int(now + self.session_seconds)
        return f"{expires}.{_sign(self.secret, str(expires))}"

    def accepts(self, token: str | None, *, now: float | None = None) -> bool:
        if not self.enabled:
            return True
        if not token:
            return False
        now = time.time() if now is None else now
        expires, _, signature = token.partition(".")
        if not signature:
            return False
        if not hmac.compare_digest(signature, _sign(self.secret, expires)):
            return False
        try:
            return int(expires) > now
        except ValueError:
            return False
