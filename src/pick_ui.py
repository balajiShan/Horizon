"""Local web UI for the pick workflow: one list of articles across all runs.

    uv run --locked python -m src.pick_ui            # http://127.0.0.1:8765

Security notes: binds to 127.0.0.1 only; rejects requests whose Host header is not
a loopback name (DNS rebinding); state-changing POSTs require a custom header and
JSON body, so other websites cannot trigger them from a browser; all article text
is untrusted and is rendered with textContent, never as HTML.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from .mcp.errors import HorizonMcpError
from .mcp.horizon_adapter import VALID_SOURCES
from .pick import PickService

_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
_LANG_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
_STATUSES = {"all", "unscored", "scored", "stale"}
_ARCHIVED = {"no", "yes", "any"}

_lock = asyncio.Lock()  # one fetch/AI job at a time: prevents double-click double spend
_service: PickService | None = None


def service() -> PickService:
    global _service
    if _service is None:
        _service = PickService()
    return _service


def _error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _host_ok(request: Request) -> bool:
    host = request.headers.get("host", "")
    name = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
    return name in _ALLOWED_HOSTS


async def _guard(request: Request) -> JSONResponse | None:
    if not _host_ok(request):
        return _error("Forbidden host", 403)
    if request.method == "POST":
        if request.headers.get("x-horizon-ui") != "1":
            return _error("Missing UI header", 403)
        if "application/json" not in request.headers.get("content-type", ""):
            return _error("JSON body required", 415)
    return None


def _seqs(value: Any) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) > 2000
        or not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in value)
    ):
        raise ValueError("Select at least one item")
    return sorted(set(value))


def _language(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not 1 <= len(value) <= 16 or not set(value) <= _LANG_OK:
        raise ValueError("Invalid language")
    return value


async def index(request: Request) -> HTMLResponse:
    if await _guard(request) is not None:
        return HTMLResponse("Forbidden", status_code=403)
    return HTMLResponse(PAGE)


async def items(request: Request) -> JSONResponse:
    if (blocked := await _guard(request)) is not None:
        return blocked
    q = request.query_params
    status = q.get("status", "all")
    sort = q.get("sort", "newest")
    source = q.get("source") or None
    archived = q.get("archived", "no")
    if status not in _STATUSES or sort not in {"newest", "oldest", "score"} or archived not in _ARCHIVED:
        return _error("Bad filter")
    try:
        days = int(q["days"]) if q.get("days") else None
    except ValueError:
        return _error("Bad days")
    tags = [t for t in q.getlist("tag") if t.strip()]
    if len(tags) > 5 or any(len(t) > 64 for t in tags):
        return _error("Too many or too long tag filters")
    signature = service().signature()
    base = dict(
        signature=signature, status=status, archived=archived, source=source, days=days,
        query=(q.get("q") or "")[:200] or None,
    )
    rows = service().store.list_items(**base, tags=tags, sort=sort, limit=1000)
    facets = service().store.tag_facets(**base)
    sources = rows[0]["all_sources"] if rows else []
    total = rows[0]["total_in_store"] if rows else 0
    for row in rows:
        row.pop("all_sources", None)
        row.pop("total_in_store", None)
    return JSONResponse({"items": rows, "sources": sources, "total": total, "tag_facets": facets, "counts": service().store.archive_counts()})


async def digest(request: Request) -> JSONResponse:
    if (blocked := await _guard(request)) is not None:
        return blocked
    try:
        name = request.path_params["name"]
        return JSONResponse({"markdown": service().read_digest(name), "data": service().read_digest_data(name)})
    except (ValueError, FileNotFoundError) as exc:
        return _error(str(exc), 404)


async def sources(request: Request) -> JSONResponse:
    if (blocked := await _guard(request)) is not None:
        return blocked
    try:
        return JSONResponse({"sources": service().source_options()})
    except Exception as exc:
        return _error(f"{type(exc).__name__}: {exc}", 500)


async def get_config(request: Request) -> JSONResponse:
    if (blocked := await _guard(request)) is not None:
        return blocked
    try:
        return JSONResponse(service().read_config_text())
    except Exception as exc:
        return _error(f"{type(exc).__name__}: {exc}", 500)


async def _job(request: Request, work) -> JSONResponse:
    if (blocked := await _guard(request)) is not None:
        return blocked
    if _lock.locked():
        return _error("Another job is still running. Wait for it to finish.", 409)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Bad request body")
    except ValueError as exc:
        return _error(str(exc))
    async with _lock:
        try:
            return JSONResponse(await work(body))
        except (ValueError, HorizonMcpError) as exc:
            return _error(str(getattr(exc, "message", exc)))
        except Exception as exc:  # surface provider/network errors in the UI
            return _error(f"{type(exc).__name__}: {exc}", 500)


async def archive(request: Request) -> JSONResponse:
    """Hide or restore articles. A quick local DB change, so it does not wait for fetch/AI jobs."""
    if (blocked := await _guard(request)) is not None:
        return blocked
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Bad request body")
        count = service().store.set_archived(_seqs(body.get("seqs")), bool(body.get("archived", True)))
    except ValueError as exc:
        return _error(str(exc))
    return JSONResponse({"changed": count})


async def save_config(request: Request) -> JSONResponse:
    async def work(body: dict) -> dict:
        text = body.get("text")
        if not isinstance(text, str):
            raise ValueError("Missing config text")
        if body.get("dry_run"):
            service().check_config_text(text)
            return {"valid": True}
        return service().save_config_text(text)

    return await _job(request, work)


async def fetch(request: Request) -> JSONResponse:
    async def work(body: dict) -> dict:
        hours = body.get("hours", 24)
        if isinstance(hours, bool) or not isinstance(hours, int) or not 1 <= hours <= 168:
            raise ValueError("hours must be 1-168")
        sources = body.get("sources")
        # An empty list would mean "everything" downstream, so it is rejected rather than defaulted.
        if not isinstance(sources, list) or not sources or not all(isinstance(s, str) for s in sources) \
                or not set(sources) <= VALID_SOURCES:
            raise ValueError("Choose at least one valid source.")
        return await service().fetch_to_store(hours, sorted(set(sources)))

    return await _job(request, work)


async def score(request: Request) -> JSONResponse:
    async def work(body: dict) -> dict:
        return await service().score_selected(_seqs(body.get("seqs")), bool(body.get("force")))

    return await _job(request, work)


async def summarize(request: Request) -> JSONResponse:
    async def work(body: dict) -> dict:
        return await service().summarize_selected(
            _seqs(body.get("seqs")), _language(body.get("language")), bool(body.get("enrich"))
        )

    return await _job(request, work)


app = Starlette(
    routes=[
        Route("/", index),
        Route("/api/items", items),
        Route("/api/digest/{name}", digest),
        Route("/api/sources", sources),
        Route("/api/config", get_config),
        Route("/api/config", save_config, methods=["POST"]),
        Route("/api/archive", archive, methods=["POST"]),
        Route("/api/fetch", fetch, methods=["POST"]),
        Route("/api/score", score, methods=["POST"]),
        Route("/api/summarize", summarize, methods=["POST"]),
    ]
)

PAGE = (Path(__file__).with_name("pick_ui.html")).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web UI for horizon-pick")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"Horizon pick UI: http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
