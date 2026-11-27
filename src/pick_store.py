"""SQLite store for the pick workflow: one row per article, across all runs.

Items are keyed by Horizon's stable item id and also deduplicated by normalised
URL, so the same article fetched in several runs (or via several sources) is one
row with one score. `seq` is a small integer that never changes and is what the
CLI and UI use to select items.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import normalize_tag, normalize_tags
from .orchestrator import _deduplication_url_key

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    id           TEXT NOT NULL UNIQUE,
    url_key      TEXT NOT NULL,
    source       TEXT NOT NULL,
    title        TEXT NOT NULL,
    url          TEXT NOT NULL,
    published_at TEXT,
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL,
    run_id       TEXT,
    item_json    TEXT NOT NULL,
    scored_json  TEXT,
    score        REAL,
    reason       TEXT,
    scored_at    TEXT,
    scored_sig   TEXT,
    archived     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS items_url_key ON items(url_key);
CREATE INDEX IF NOT EXISTS items_first_seen ON items(first_seen);
"""

SORTS = {
    "newest": "first_seen DESC, seq DESC",
    "oldest": "first_seen ASC, seq ASC",
    "score": "score IS NULL, score DESC, seq DESC",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def url_key(url: str) -> str:
    return json.dumps(_deduplication_url_key(url))


class PickStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(items)")}
            if "archived" not in columns:  # database created before archiving existed
                conn.execute("ALTER TABLE items ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
            self._canonicalize_stored_tags(conn)

    @staticmethod
    def _canonicalize_stored_tags(conn: sqlite3.Connection) -> int:
        """Rewrite tags of already-scored items to the canonical form (idempotent)."""
        changed = 0
        for row in conn.execute("SELECT seq, scored_json FROM items WHERE scored_json IS NOT NULL").fetchall():
            data = json.loads(row["scored_json"])
            analysis = (data.get("processing") or {}).get("analysis")
            tags = analysis.get("tags") if isinstance(analysis, dict) else None
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                continue
            fixed = normalize_tags(tags)
            if fixed != tags:
                analysis["tags"] = fixed
                conn.execute("UPDATE items SET scored_json = ? WHERE seq = ?", (json.dumps(data), row["seq"]))
                changed += 1
        return changed

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.create_function("norm_tag", 1, lambda v: normalize_tag(v) if v is not None else "", deterministic=True)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # -- writing ---------------------------------------------------------------

    def upsert_items(
        self, items: list[dict[str, Any]], run_id: str | None, seen_at: str | None = None
    ) -> dict[str, int]:
        """Insert unseen items. Known ids / URLs only get their last_seen refreshed."""
        added = duplicates = 0
        now = seen_at or _now()
        with self._conn() as conn:
            for item in items:
                key = url_key(str(item["url"]))
                seen = conn.execute(
                    "SELECT seq FROM items WHERE id = ? OR url_key = ? LIMIT 1", (item["id"], key)
                ).fetchone()
                if seen:
                    conn.execute("UPDATE items SET last_seen = ? WHERE seq = ?", (now, seen["seq"]))
                    duplicates += 1
                    continue
                conn.execute(
                    "INSERT INTO items (id, url_key, source, title, url, published_at, first_seen,"
                    " last_seen, run_id, item_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        item["id"], key, item["source_type"], item["title"], str(item["url"]),
                        item.get("published_at"), now, now, run_id,
                        json.dumps({**item, "processing": None}),
                    ),
                )
                added += 1
        return {"added": added, "duplicates": duplicates}

    def save_scored(self, item_id: str, scored: dict[str, Any], signature: str) -> None:
        analysis = ((scored.get("processing") or {}).get("analysis")) or {}
        with self._conn() as conn:
            conn.execute(
                "UPDATE items SET scored_json = ?, score = ?, reason = ?, scored_at = ?, scored_sig = ?"
                " WHERE id = ?",
                (json.dumps(scored), analysis.get("score"), analysis.get("reason"), _now(), signature, item_id),
            )

    def set_archived(self, seqs: list[int], archived: bool) -> int:
        if not seqs:
            return 0
        marks = ",".join("?" * len(seqs))
        with self._conn() as conn:
            cur = conn.execute(f"UPDATE items SET archived = ? WHERE seq IN ({marks})", [int(archived), *seqs])
            return cur.rowcount

    def archive_counts(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute("SELECT archived, COUNT(*) FROM items GROUP BY archived").fetchall()
        counts = {int(flag): n for flag, n in rows}
        return {"active": counts.get(0, 0), "archived": counts.get(1, 0)}

    def delete_unscored_older_than(self, days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM items WHERE scored_json IS NULL AND first_seen < ?", (cutoff,))
            return cur.rowcount

    # -- reading ---------------------------------------------------------------

    def get_many(self, seqs: list[int]) -> list[sqlite3.Row]:
        if not seqs:
            return []
        marks = ",".join("?" * len(seqs))
        with self._conn() as conn:
            rows = conn.execute(f"SELECT * FROM items WHERE seq IN ({marks}) ORDER BY seq", seqs).fetchall()
        return rows

    @staticmethod
    def norm_tag(tag: str) -> str:
        return normalize_tag(tag)

    def _filters(
        self, signature: str, status: str, archived: str, source: str | None, days: int | None, query: str | None
    ) -> tuple[list[str], list[Any]]:
        where: list[str] = []
        args: list[Any] = []
        if status == "unscored":
            where.append("scored_json IS NULL")
        elif status == "scored":
            where.append("scored_json IS NOT NULL AND scored_sig = ?")
            args.append(signature)
        elif status == "stale":
            where.append("scored_json IS NOT NULL AND scored_sig IS NOT ?")
            args.append(signature)
        if archived == "no":
            where.append("archived = 0")
        elif archived == "yes":
            where.append("archived = 1")
        if source:
            where.append("source = ?")
            args.append(source)
        if days:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
            where.append("first_seen >= ?")
            args.append(cutoff)
        if query:
            where.append("title LIKE ? ESCAPE '\\'")
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            args.append(f"%{escaped}%")
        return where, args

    def list_items(
        self,
        *,
        signature: str,
        status: str = "all",
        archived: str = "no",
        source: str | None = None,
        days: int | None = None,
        query: str | None = None,
        tags: list[str] | None = None,
        sort: str = "newest",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        where, args = self._filters(signature, status, archived, source, days, query)
        for tag in tags or []:  # every chosen tag must be present (AND)
            where.append(
                "EXISTS (SELECT 1 FROM json_each(items.scored_json, '$.processing.analysis.tags') AS t"
                " WHERE norm_tag(t.value) = ?)"
            )
            args.append(self.norm_tag(tag))
        order = SORTS.get(sort, SORTS["newest"])
        sql = "SELECT * FROM items" + (" WHERE " + " AND ".join(where) if where else "")
        sql += f" ORDER BY {order} LIMIT ?"  # `order` comes from the fixed SORTS map
        with self._conn() as conn:
            rows = conn.execute(sql, [*args, int(limit)]).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            sources = [r[0] for r in conn.execute("SELECT DISTINCT source FROM items ORDER BY source")]
        out = [self.to_view(row, signature) for row in rows]
        for entry in out:
            entry["total_in_store"] = total
            entry["all_sources"] = sources
        return out

    def tag_facets(
        self,
        *,
        signature: str,
        status: str = "all",
        archived: str = "no",
        source: str | None = None,
        days: int | None = None,
        query: str | None = None,
        limit: int = 24,
    ) -> list[dict[str, Any]]:
        """Most common tags among the articles matching the other filters (tag filters excluded)."""
        where, args = self._filters(signature, status, archived, source, days, query)
        where.append("scored_json IS NOT NULL")
        sql = "SELECT scored_json FROM items WHERE " + " AND ".join(where)
        counts: dict[str, int] = {}
        labels: dict[str, dict[str, int]] = {}
        with self._conn() as conn:
            for (blob,) in conn.execute(sql, args):
                tags = ((json.loads(blob).get("processing") or {}).get("analysis") or {}).get("tags") or []
                for tag in {self.norm_tag(str(t)): self.norm_tag(str(t)) for t in tags}.items():
                    key, shown = tag
                    if not key:
                        continue
                    counts[key] = counts.get(key, 0) + 1
                    labels.setdefault(key, {})[shown] = labels.setdefault(key, {}).get(shown, 0) + 1
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return [{"tag": max(labels[k], key=labels[k].get), "key": k, "count": c} for k, c in top]

    @staticmethod
    def to_view(row: sqlite3.Row, signature: str) -> dict[str, Any]:
        scored = row["scored_json"] is not None
        state = "unscored" if not scored else ("scored" if row["scored_sig"] == signature else "stale")
        url = row["url"] if row["url"].startswith(("http://", "https://")) else ""

        item = json.loads(row["item_json"])
        meta = item.get("metadata") or {}
        analysis = ((json.loads(row["scored_json"]).get("processing") or {}).get("analysis") or {}) if scored else {}
        return {
            "seq": row["seq"],
            "source": row["source"],
            "feed": meta.get("feed_name") or meta.get("subreddit") or "",
            "title": row["title"],
            "url": url,
            "published_at": row["published_at"],
            "first_seen": row["first_seen"],
            "state": state,
            "archived": bool(row["archived"]),
            "score": row["score"],
            "reason": row["reason"],
            "summary": analysis.get("summary") or "",
            "tags": normalize_tags(analysis.get("tags") or []),
            "excerpt": _excerpt(item.get("content")),
        }


_TAG = re.compile(r"<[^>]+>")


def _excerpt(content: str | None, limit: int = 240) -> str:
    """Plain-text teaser of the fetched content (shown for articles not scored yet)."""
    if not content:
        return ""
    text = " ".join(_TAG.sub(" ", content).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"
