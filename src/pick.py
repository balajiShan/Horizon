"""Pick-first workflow across all runs: fetch titles, score chosen items, summarize chosen items.

    horizon-pick fetch --hours 24          # fetch, add new articles to the store, list them
    horizon-pick list --status unscored    # browse the store (all runs together)
    horizon-pick score 12,15-18            # AI-score the chosen items (skips already scored)
    horizon-pick summarize 12,15           # digest from only the chosen (scored) items
    horizon-pick import-runs               # one-off: import older data/mcp-runs/* into the store

Item numbers are permanent store ids, not positions: #12 is the same article in every run.
Every article is stored once (deduplicated by item id and normalised URL), with its score
attached, so nothing is scored or paid for twice.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from ._cli import add_log_level_argument
from .ai.summarizer import DailySummarizer
from .mcp.errors import HorizonMcpError
from .mcp.horizon_adapter import dicts_to_items, get_enabled_sources, items_to_dicts
from .mcp.service import HorizonPipelineService
from .models import SOURCE_REGISTRY, AIProvider, Config
from .storage.manager import _expand_env_vars
from .pick_store import PickStore

out = Console()
_DIGEST_NAME = re.compile(r"^digest-[0-9T-]+-[A-Za-z0-9_-]{1,16}\.md$")


def parse_selection(spec: str) -> list[int]:
    """Parse '1,3,5-7' into sorted unique positive ints."""
    numbers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
            start, end = int(low), int(high)
            if start > end or end - start > 5000:
                raise ValueError(f"Bad range: {part}")
            numbers.update(range(start, end + 1))
        else:
            numbers.add(int(part))
    if not numbers or min(numbers) < 1:
        raise ValueError(f"Bad selection: {spec!r}")
    return sorted(numbers)


class PickService(HorizonPipelineService):
    """Staged service plus a cross-run item store with per-item scoring."""

    def __init__(self, runs_root: Path | None = None, **kwargs: Any):
        super().__init__(runs_root=runs_root, **kwargs)
        self.data_dir = self.runs_root.parent
        self.store = PickStore(self.data_dir / "pick.db")
        self.digest_dir = self.data_dir / "pick-summaries"

    # -- scoring signature: changes when the model or any profile rubric changes ----------

    def signature(self, ctx: Any = None) -> str:
        ctx = ctx or self._build_context(horizon_path=None, config_path=None, sources=None)[0]
        digest = hashlib.sha256()
        digest.update(f"{ctx.config.ai.provider}|{ctx.config.ai.model}".encode())
        profiles = Path(ctx.config.processing.profiles_dir).expanduser()
        if not profiles.is_absolute():
            profiles = ctx.horizon_path / profiles
        for path in sorted(profiles.glob("*/analysis.md")):
            digest.update(path.parent.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()[:16]

    # -- operations ----------------------------------------------------------------------

    async def fetch_to_store(self, hours: int, sources: list[str] | None) -> dict[str, Any]:
        result = await self.fetch_items(hours=hours, sources=sources)
        run_id = result["run_id"]
        counts = {"added": 0, "duplicates": 0}
        if result["fetched"]:
            counts = self.store.upsert_items(self.run_store.load_items(run_id, "raw"), run_id)
        report = result.get("fetch_report") or {}
        failures = [
            f"{s['source']}: {s['status']}"
            for s in report.get("sources", [])
            if s.get("status") in {"failure", "partial_failure"}
        ]
        return {"run_id": run_id, "fetched": result["fetched"], **counts, "failures": failures}

    def import_runs(self) -> dict[str, int]:
        """Import raw (and scored) items from older runs; keeps scores you already paid for."""
        signature = self.signature()
        imported = scores = 0
        for run in sorted(self.run_store.list_runs(limit=1000), key=lambda r: r.get("created_at") or ""):
            run_id = run["run_id"]
            if not self.run_store.has_stage(run_id, "raw"):
                continue
            imported += self.store.upsert_items(
                self.run_store.load_items(run_id, "raw"), run_id, seen_at=run.get("created_at")
            )["added"]
            if self.run_store.has_stage(run_id, "scored"):
                for item in self.run_store.load_items(run_id, "scored"):
                    analysis = ((item.get("processing") or {}).get("analysis")) or {}
                    if analysis.get("score") is not None:
                        self.store.save_scored(item["id"], item, signature)
                        scores += 1
        return {"items_added": imported, "scores_imported": scores}

    async def score_selected(self, seqs: list[int], force: bool = False) -> dict[str, Any]:
        rows = self.store.get_many(seqs)
        missing = sorted(set(seqs) - {r["seq"] for r in rows})
        if missing:
            raise ValueError(f"Unknown item number(s): {missing}")

        ctx, _, _ = self._build_context(horizon_path=None, config_path=None, sources=None)
        signature = self.signature(ctx)
        todo = [r for r in rows if force or r["scored_json"] is None or r["scored_sig"] != signature]
        skipped = len(rows) - len(todo)
        if not todo:
            return {"scored_now": 0, "skipped": skipped}

        if ctx.config.ai.provider == AIProvider.OPENAI_CHATGPT and todo[0]["run_id"]:
            self._bind_chatgpt_account(ctx, todo[0]["run_id"])
        import json

        items = dicts_to_items([json.loads(r["item_json"]) for r in todo])
        scored = await self._orchestrator(ctx).analyze_items(items)
        for item, dumped in zip(scored, items_to_dicts(scored)):
            self.store.save_scored(item.id, dumped, signature)
        return {"scored_now": len(scored), "skipped": skipped}

    async def summarize_selected(
        self, seqs: list[int], language: str | None, enrich: bool
    ) -> dict[str, Any]:
        import json

        rows = self.store.get_many(seqs)
        missing = sorted(set(seqs) - {r["seq"] for r in rows})
        if missing:
            raise ValueError(f"Unknown item number(s): {missing}")
        unscored = [r["seq"] for r in rows if r["scored_json"] is None]
        if unscored:
            raise ValueError(f"Not scored yet: {unscored}. Score them first.")

        ctx, _, _ = self._build_context(horizon_path=None, config_path=None, sources=None)
        items = dicts_to_items([json.loads(r["scored_json"]) for r in rows])
        if enrich:
            result = await self._orchestrator(ctx).enrich_items(items)
            if result.status == "failure":
                out.print("[yellow]Enrichment failed; summarizing without it.[/yellow]")

        language = language or ctx.config.ai.languages[0]
        summarizer = DailySummarizer(
            profile_names=self._profiles(ctx).names,
            profile_order=ctx.config.digest.profile_order,
        )
        now = datetime.now(timezone.utc)
        text = await summarizer.generate_summary(
            items, now.strftime("%Y-%m-%d"), len(items), language=language
        )
        self.digest_dir.mkdir(parents=True, exist_ok=True)
        name = f"digest-{now.strftime('%Y%m%dT%H%M%S')}-{language}.md"
        (self.digest_dir / name).write_text(text, encoding="utf-8")
        self._write_digest_data(name, rows, items, language, now)
        return {"name": name, "path": str((self.digest_dir / name).resolve()), "items": len(items), "language": language}

    def trace(self, seq: int) -> list[dict[str, str]]:
        """Why this article is in the store: the chain of gates it passed.

        Steps are derived from the stored item, your current config and (if scored) the saved AI
        analysis. They describe the rules in force now, not a log recorded at fetch time.
        """
        import json

        rows = self.store.get_many([seq])
        if not rows:
            raise ValueError(f"Unknown item number: {seq}")
        row = rows[0]
        item = json.loads(row["item_json"])
        meta = item.get("metadata") or {}
        ctx, _, _ = self._build_context(horizon_path=None, config_path=None, sources=None)
        source = item["source_type"]
        steps: list[dict[str, str]] = []

        where = meta.get("feed_name") or meta.get("subreddit") or ("front page" if source == "hackernews" else "")
        steps.append({"step": "1. Source", "result": "in", "detail": f"{source}" + (f" / {where}" if where else "") + " is one of your enabled sources."})

        hn = getattr(ctx.config.sources, "hackernews", None)
        if source == "hackernews" and hn is not None:
            steps.append({
                "step": "2. Fetch gate", "result": "pass",
                "detail": f"Rule: top {hn.fetch_top_stories} Hacker News stories with at least {hn.min_score} points. "
                          f"This one had {meta.get('score')} points and {meta.get('descendants')} comments.",
            })
        elif source == "rss":
            steps.append({
                "step": "2. Fetch gate", "result": "pass",
                "detail": "Rule: every new entry in the fetch window. There is no topic or keyword filter on RSS, "
                          "so a broad feed (e.g. an aggregator) pulls everything it publishes.",
            })
        else:
            steps.append({"step": "2. Fetch gate", "result": "pass", "detail": f"Collected by the {source} scraper; see its settings in config.json."})

        steps.append({
            "step": "3. Dedup", "result": "pass",
            "detail": f"First seen {row['first_seen']} (run {row['run_id'] or 'n/a'}). No earlier stored article has the same id or normalised URL.",
        })

        if row["scored_json"] is None:
            steps.append({"step": "4. AI score", "result": "pending", "detail": "Not scored yet. No AI call has been made for this article."})
            return steps

        scored = json.loads(row["scored_json"])
        processing = scored.get("processing") or {}
        analysis = processing.get("analysis") or {}
        classification = processing.get("classification") or {}
        profile = classification.get("profile", "?")
        how = classification.get("method", "?")
        steps.append({
            "step": "4. AI score", "result": "scored",
            "detail": f"Profile {profile} ({how}). Score {analysis.get('score')}: {analysis.get('reason')}",
        })
        settings = ctx.config.processing.profile_settings.get(profile)
        threshold = getattr(settings, "threshold", None)
        score = analysis.get("score")
        if threshold is None or score is None:
            steps.append({"step": "5. Digest threshold", "result": "unknown", "detail": "No threshold configured for this profile."})
        else:
            ok = score >= threshold
            steps.append({
                "step": "5. Digest threshold", "result": "pass" if ok else "fail",
                "detail": f"{profile} needs {threshold}; this scored {score}. "
                          + ("It would make the normal daily digest." if ok else "The normal daily digest would drop it."),
            })
        return steps

    def _write_digest_data(self, name: str, rows: list[Any], items: list[Any], language: str, now: datetime) -> None:
        """Structured copy of a digest so the UI can render cards instead of raw markdown."""
        import json

        entries = []
        for row, item in zip(rows, items):
            analysis = item.processing.analysis if item.processing and item.processing.analysis else None
            url = str(item.url)
            entries.append(
                {
                    "seq": row["seq"],
                    "title": item.title,
                    "url": url if url.startswith(("http://", "https://")) else "",
                    "source": item.source_type.value,
                    "feed": (item.metadata or {}).get("feed_name") or "",
                    "published_at": item.published_at.isoformat() if item.published_at else None,
                    "score": analysis.score if analysis else None,
                    "summary": analysis.summary if analysis else "",
                    "reason": analysis.reason if analysis else "",
                    "tags": list(analysis.tags) if analysis else [],
                }
            )
        data = {"name": name, "created_at": now.isoformat(timespec="seconds"), "language": language, "items": entries}
        (self.digest_dir / name.replace(".md", ".json")).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def read_digest_data(self, name: str) -> dict[str, Any] | None:
        import json

        if not _DIGEST_NAME.match(name):
            raise ValueError("Invalid digest name")
        path = self.digest_dir / name.replace(".md", ".json")
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def source_options(self, config: Config | None = None) -> list[dict[str, Any]]:
        """Every source type Horizon supports, with whether it is enabled in your config."""
        config = config or self._build_context(horizon_path=None, config_path=None, sources=None)[0].config
        enabled = set(get_enabled_sources(config))
        options = []
        for name, definition in SOURCE_REGISTRY.items():
            section = getattr(config.sources, definition.config_field, None)
            if definition.config_is_list:
                active = sum(1 for entry in section or [] if getattr(entry, "enabled", True))
                detail = f"{active} configured" if active else ""
            else:
                entries = sum(len(getattr(section, f, None) or []) for f in definition.item_fields)
                detail = f"{entries} configured" if entries else ""
            options.append({"id": name, "enabled": name in enabled, "detail": detail})
        return options

    # -- configuration editor -------------------------------------------------------------

    MAX_CONFIG_BYTES = 1_000_000
    KEEP_BACKUPS = 20

    def config_path(self) -> Path:
        return self._build_context(horizon_path=None, config_path=None, sources=None)[0].config_path

    def read_config_text(self) -> dict[str, str]:
        path = self.config_path()
        return {"path": str(path), "text": path.read_text(encoding="utf-8")}

    @staticmethod
    def check_config_text(text: str) -> dict[str, Any]:
        """Parse and validate config text the same way Horizon loads it. Raises ValueError."""
        if len(text.encode("utf-8")) > PickService.MAX_CONFIG_BYTES:
            raise ValueError("Config is too large.")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno}).") from exc
        if not isinstance(parsed, dict):
            raise ValueError("The config must be a JSON object.")
        try:
            Config.model_validate(_expand_env_vars(parsed))
        except Exception as exc:
            errors = getattr(exc, "errors", None)
            if callable(errors):
                lines = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in errors()[:8]]
                raise ValueError("Config rejected by Horizon:\n" + "\n".join(lines)) from exc
            raise ValueError(f"Config rejected by Horizon: {exc}") from exc
        return parsed

    def save_config_text(self, text: str) -> dict[str, Any]:
        """Validate, back up the current file, then replace it atomically."""
        parsed = self.check_config_text(text)
        path = self.config_path()
        backups = self.data_dir / "config-backups"
        backups.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup = backups / f"config-{stamp}.json"
        if path.exists():
            backup.write_bytes(path.read_bytes())

        body = json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"
        fd, tmp = tempfile.mkstemp(prefix=".config.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(body)
            os.chmod(tmp, path.stat().st_mode & 0o777 if path.exists() else 0o600)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

        for old in sorted(backups.glob("config-*.json"))[: -self.KEEP_BACKUPS]:
            old.unlink(missing_ok=True)
        return {"saved": True, "backup": backup.name if backup.exists() else None, "text": body}

    def read_digest(self, name: str) -> str:
        if not _DIGEST_NAME.match(name):
            raise ValueError("Invalid digest name")
        return (self.digest_dir / name).read_text(encoding="utf-8")


def print_items(rows: list[dict[str, Any]]) -> None:
    table = Table(show_lines=False)
    for col, kwargs in (("#", {"justify": "right"}), ("Seen", {}), ("Source", {}), ("Title", {"overflow": "fold"}), ("Score", {"justify": "right"})):
        table.add_column(col, **kwargs)
    for row in rows:
        score = "-" if row["state"] == "unscored" else ("n/a" if row["score"] is None else f"{row['score']:.1f}")
        if row["state"] == "stale":
            score += "*"
        table.add_row(str(row["seq"]), row["first_seen"][5:16].replace("T", " "), row["source"], row["title"], score)
    out.print(table)
    if any(r["state"] == "stale" for r in rows):
        out.print("[dim]* stale: scored with an older model or rubric[/dim]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="horizon-pick", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_log_level_argument(parser)
    parser.add_argument("--runs-root", help="Run store directory (default: data/mcp-runs)")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Fetch sources and add new articles to the store (no AI)")
    fetch.add_argument("--hours", type=int, default=24)
    fetch.add_argument("--sources", help="Comma-separated, e.g. hackernews,rss")

    lst = sub.add_parser("list", help="List stored articles across all runs")
    lst.add_argument("--status", choices=["all", "unscored", "scored", "stale"], default="all")
    lst.add_argument("--source")
    lst.add_argument("--days", type=int, help="Only items first seen in the last N days")
    lst.add_argument("--search", help="Title contains")
    lst.add_argument("--sort", choices=["newest", "oldest", "score"], default="newest")
    lst.add_argument("--limit", type=int, default=100)

    score = sub.add_parser("score", help="AI-score only the chosen items")
    score.add_argument("selection", help="Item numbers, e.g. 12,15-18")
    score.add_argument("--force", action="store_true", help="Re-score even if already scored")

    summ = sub.add_parser("summarize", help="Digest from only the chosen (already scored) items")
    summ.add_argument("selection", help="Item numbers, e.g. 12,15")
    summ.add_argument("--language", help="Output language (default: first of ai.languages)")
    summ.add_argument("--enrich", action="store_true", help="Add background enrichment (extra AI + web search)")

    sub.add_parser("import-runs", help="Import older data/mcp-runs/* into the store")
    return parser


async def run(args: argparse.Namespace) -> None:
    service = PickService(runs_root=Path(args.runs_root) if args.runs_root else None)

    if args.command == "fetch":
        r = await service.fetch_to_store(args.hours, [s.strip() for s in args.sources.split(",")] if args.sources else None)
        out.print(f"Fetched {r['fetched']}: {r['added']} new, {r['duplicates']} already in store.")
        for failure in r["failures"]:
            out.print(f"[yellow]{failure}[/yellow]")
        print_items(service.store.list_items(signature=service.signature(), status="unscored", limit=50))
    elif args.command == "list":
        rows = service.store.list_items(
            signature=service.signature(), status=args.status, archived="any", source=args.source, days=args.days,
            query=args.search, sort=args.sort, limit=args.limit,
        )
        print_items(rows)
        out.print(f"{len(rows)} shown, {rows[0]['total_in_store'] if rows else 0} in store.")
    elif args.command == "score":
        r = await service.score_selected(parse_selection(args.selection), args.force)
        out.print(f"Scored {r['scored_now']}, skipped {r['skipped']} already scored (use --force to redo).")
    elif args.command == "summarize":
        r = await service.summarize_selected(parse_selection(args.selection), args.language, args.enrich)
        out.print(f"Digest of {r['items']} item(s), language={r['language']}: {r['path']}")
    elif args.command == "import-runs":
        out.print(service.import_runs())


def main() -> None:
    args = build_parser().parse_args()
    import logging

    logging.basicConfig(level=args.log_level)
    try:
        asyncio.run(run(args))
    except (HorizonMcpError, ValueError) as exc:
        out.print(f"[red]{getattr(exc, 'message', exc)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
