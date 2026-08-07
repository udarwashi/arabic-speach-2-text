"""Static checks on the page and its script.

There is no browser in the test environment, so instead of rendering the page we
verify the things that silently break it: an element id the script reaches for
but the markup does not define, a missing RTL attribute, or a reference to an
asset that is not served.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
JS = (STATIC / "app.js").read_text(encoding="utf-8")
CSS = (STATIC / "styles.css").read_text(encoding="utf-8")

HTML_IDS = set(re.findall(r'\bid="([^"]+)"', HTML))
JS_IDS = set(re.findall(r"\bel\('([^']+)'\)", JS))


def test_every_id_the_script_uses_exists_in_the_markup() -> None:
    assert JS_IDS, "the id extraction regex found nothing — check the test"
    assert JS_IDS <= HTML_IDS, f"missing from index.html: {sorted(JS_IDS - HTML_IDS)}"


def test_the_document_is_right_to_left_arabic() -> None:
    assert '<html lang="ar" dir="rtl">' in HTML
    assert '<meta charset="utf-8" />' in HTML


def test_the_transcript_panes_declare_their_direction() -> None:
    # Explicit per-element direction, so the transcript stays RTL even if the
    # page is later embedded somewhere LTR.
    for pane in ('id="transcript-flow"', 'id="transcript-timed"'):
        block = HTML[HTML.index(pane) - 200 : HTML.index(pane) + 200]
        assert 'dir="rtl"' in block and 'lang="ar"' in block


def test_timestamps_are_bidi_isolated() -> None:
    """Latin digits inside an Arabic run reorder visually without isolation."""
    assert "unicode-bidi: isolate" in CSS
    assert re.search(r"\.ts\s*\{[^}]*direction:\s*ltr", CSS, re.S)
    assert "stamp.dir = 'ltr'" in JS


def test_the_layout_uses_logical_properties_not_physical_sides() -> None:
    physical = re.findall(r"\b(margin|padding)-(left|right)\s*:", CSS)
    assert not physical, f"physical side properties break RTL mirroring: {physical}"


def test_the_page_loads_no_external_resources() -> None:
    """Everything must be local: the app is usable with no network."""
    external = re.findall(r'(?:href|src)="(https?://[^"]+)"', HTML)
    assert not external, f"external resources found: {external}"
    assert "@import" not in CSS
    assert "fonts.googleapis" not in CSS


@pytest.mark.parametrize("asset", ["/static/styles.css", "/static/app.js", "/static/logo.svg"])
def test_referenced_assets_are_actually_served(client, asset: str) -> None:
    assert asset in HTML
    assert client.get(asset).status_code == 200


def test_every_bundled_font_is_served(client) -> None:
    """The fonts are vendored, so a missing file silently degrades the page."""
    urls = re.findall(r'url\("(/static/fonts/[^"]+)"\)', CSS)
    assert urls, "no @font-face sources found — check the test"
    for url in set(urls):
        assert client.get(url).status_code == 200, url


def test_the_script_only_calls_endpoints_that_exist(client) -> None:
    served = {route.path for route in client.app.routes}
    # Literal paths, plus the templated ones reduced to their route pattern.
    used = set(re.findall(r"fetch\('(/api/[a-z]+)'", JS))
    used |= {
        "/api/jobs/{job_id}"
        for _ in re.findall(r"/api/jobs/\$\{jobId\}`", JS)
    }
    used |= {
        "/api/jobs/{job_id}/stream"
        for _ in re.findall(r"/api/jobs/\$\{jobId\}/stream", JS)
    }
    assert used, "no endpoint references found — check the test"
    assert used <= served, f"script calls unserved routes: {sorted(used - served)}"


def test_the_stream_is_closed_on_every_terminal_event() -> None:
    """An EventSource left open after 'done' reconnects in a loop."""
    for event in ("done", "error", "cancelled"):
        assert f"addEventListener('{event}'" in JS
    assert "finishRun" in JS and "source.close()" in JS


def test_segments_are_deduplicated_by_index() -> None:
    """A reconnect replays the snapshot, so rendering must be idempotent."""
    assert "state.seen.has(segment.index)" in JS
    assert "state.seen.add(segment.index)" in JS


def test_no_transcript_text_is_inserted_as_html() -> None:
    """Transcribed text is untrusted input; it must go in via textContent."""
    assert "innerHTML" not in JS
