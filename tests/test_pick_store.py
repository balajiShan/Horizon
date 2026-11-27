from __future__ import annotations

import json

import pytest

from src.pick import parse_selection
from src.pick_store import PickStore


def item(item_id: str, url: str, title: str = "T", source: str = "rss") -> dict:
    return {
        "id": item_id,
        "source_type": source,
        "title": title,
        "url": url,
        "published_at": "2026-11-25T00:00:00Z",
        "metadata": {},
        "processing": None,
    }


def scored(base: dict, score: float | None) -> dict:
    return {**base, "processing": {"analysis": {"score": score, "reason": "because", "summary": "s", "tags": []}}}


@pytest.fixture()
def store(tmp_path):
    return PickStore(tmp_path / "pick.db")


def test_upsert_dedupes_across_runs_by_id_and_url(store):
    first = store.upsert_items([item("a", "https://x.com/1"), item("b", "https://x.com/2")], "run1")
    assert first == {"added": 2, "duplicates": 0}

    # Same id again, and a different id carrying the same URL (tracking params ignored).
    again = store.upsert_items(
        [item("a", "https://x.com/1"), item("c", "https://X.com/2/?utm_source=feed")], "run2"
    )
    assert again == {"added": 0, "duplicates": 2}
    assert len(store.list_items(signature="s")) == 2


def test_seq_numbers_are_stable_and_global(store):
    store.upsert_items([item("a", "https://x.com/1")], "run1")
    store.upsert_items([item("b", "https://x.com/2")], "run2")
    rows = {r["seq"]: r for r in store.get_many([1, 2])}
    assert rows[1]["id"] == "a" and rows[2]["id"] == "b"


def test_score_state_and_stale_detection(store):
    base = item("a", "https://x.com/1")
    store.upsert_items([base, item("b", "https://x.com/2")], "run1")
    store.save_scored("a", scored(base, 7.5), signature="v1")

    states = {r["seq"]: r["state"] for r in store.list_items(signature="v1")}
    assert states == {1: "scored", 2: "unscored"}

    # A changed model/rubric signature makes the old score stale, not lost.
    states = {r["seq"]: r["state"] for r in store.list_items(signature="v2")}
    assert states[1] == "stale"
    assert [r["seq"] for r in store.list_items(signature="v2", status="stale")] == [1]
    assert [r["seq"] for r in store.list_items(signature="v1", status="unscored")] == [2]
    assert store.list_items(signature="v1", status="scored")[0]["score"] == 7.5


def test_filters_sort_and_search_are_safe(store):
    store.upsert_items(
        [item("a", "https://x.com/1", "Rust 100%_fast", "rss"), item("b", "https://x.com/2", "Other", "hackernews")], "r"
    )
    store.save_scored("b", scored(item("b", "https://x.com/2"), 9.0), signature="s")
    assert [r["source"] for r in store.list_items(signature="s", source="hackernews")] == ["hackernews"]
    assert [r["seq"] for r in store.list_items(signature="s", sort="score")] == [2, 1]
    # LIKE wildcards in the search text are matched literally.
    assert [r["seq"] for r in store.list_items(signature="s", query="100%_")] == [1]
    # A lone "%" matches only the title that literally contains one, not everything.
    assert [r["seq"] for r in store.list_items(signature="s", query="%")] == [1]
    # An injection-looking value is just text.
    assert store.list_items(signature="s", query="x' OR '1'='1") == []


def test_non_http_urls_are_not_exposed_as_links(store):
    store.upsert_items([item("a", "javascript:alert(1)")], "r")
    assert store.list_items(signature="s")[0]["url"] == ""


def test_parse_selection():
    assert parse_selection("3, 1,5-7,3") == [1, 3, 5, 6, 7]
    for bad in ("", "0", "a", "5-2", "1-99999"):
        with pytest.raises(ValueError):
            parse_selection(bad)
    json.dumps(parse_selection("1"))  # plain ints, JSON-safe


def test_digest_data_sidecar_roundtrip(tmp_path):
    from datetime import datetime, timezone

    from src.models import ContentAnalysis, ContentItem, ProcessingResult, ClassificationResult
    from src.pick import PickService

    service = PickService(runs_root=tmp_path / "mcp-runs")
    item = ContentItem(
        id="rss:feed:1",
        source_type="rss",
        title="Rustc&#x27;s <b>solver</b>",
        url="https://example.com/post",
        published_at=datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc),
        metadata={"feed_name": "Lobsters"},
        processing=ProcessingResult(
            classification=ClassificationResult(profile="tech-news", method="source_override"),
            analysis=ContentAnalysis(score=7.0, reason="r", summary="sum", tags=["Rust"]),
        ),
    )
    name = "digest-20261125T120000-en.md"
    service.digest_dir.mkdir(parents=True)
    (service.digest_dir / name).write_text("# md", encoding="utf-8")
    service._write_digest_data(name, [{"seq": 4}], [item], "en", datetime(2026, 11, 25, tzinfo=timezone.utc))

    data = service.read_digest_data(name)
    entry = data["items"][0]
    assert entry["seq"] == 4 and entry["score"] == 7.0 and entry["tags"] == ["rust"]
    assert entry["feed"] == "Lobsters" and entry["title"] == "Rustc&#x27;s <b>solver</b>"  # stored raw; UI uses textContent
    assert service.read_digest_data("digest-20261125T130000-en.md") is None  # older digest, no sidecar
    with pytest.raises(ValueError):
        service.read_digest_data("../../etc/passwd")


def tagged(base: dict, tags: list[str], score: float = 5.0) -> dict:
    return {
        **base,
        "processing": {"analysis": {"score": score, "reason": "r", "summary": "sum", "tags": tags}},
    }


def test_archive_hides_and_restores_without_losing_scores(store):
    a, b = item("a", "https://x.com/1"), item("b", "https://x.com/2")
    store.upsert_items([a, b], "r")
    store.save_scored("a", tagged(a, ["Rust"]), signature="s")

    assert store.set_archived([1], True) == 1
    assert [r["seq"] for r in store.list_items(signature="s")] == [2]  # default view hides archived
    archived = store.list_items(signature="s", archived="yes")
    assert [r["seq"] for r in archived] == [1] and archived[0]["archived"] is True
    assert archived[0]["score"] == 5.0  # score kept while archived
    assert len(store.list_items(signature="s", archived="any")) == 2

    assert store.set_archived([1], False) == 1
    assert [r["seq"] for r in store.list_items(signature="s")] == [2, 1]
    assert store.set_archived([], True) == 0


def test_tag_filter_is_case_and_hash_insensitive_and_ands(store):
    a, b, c = item("a", "https://x.com/1"), item("b", "https://x.com/2"), item("c", "https://x.com/3")
    store.upsert_items([a, b, c], "r")
    store.save_scored("a", tagged(a, ["Rust", "#Performance"]), signature="s")
    store.save_scored("b", tagged(b, ["rust", "Security"]), signature="s")
    # c stays unscored: it has no tags and must never match a tag filter.

    def seqs(**kw):
        return sorted(r["seq"] for r in store.list_items(signature="s", **kw))

    assert seqs(tags=["#RUST"]) == [1, 2]
    assert seqs(tags=["rust", "performance"]) == [1]  # both tags required
    assert seqs(tags=["security"]) == [2]
    assert seqs(tags=["nonexistent"]) == []
    assert seqs(tags=["x') OR 1=1 --"]) == []  # parameterised, not interpolated


def test_tag_facets_count_across_variants_and_ignore_unscored(store):
    a, b, c = item("a", "https://x.com/1"), item("b", "https://x.com/2"), item("c", "https://x.com/3")
    store.upsert_items([a, b, c], "r")
    store.save_scored("a", tagged(a, ["Rust", "Performance"]), signature="s")
    store.save_scored("b", tagged(b, ["rust"]), signature="s")
    facets = store.tag_facets(signature="s")
    assert facets[0]["key"] == "rust" and facets[0]["count"] == 2
    assert {f["key"] for f in facets} == {"rust", "performance"}
    store.set_archived([2], True)
    assert [f["count"] for f in store.tag_facets(signature="s") if f["key"] == "rust"] == [1]


def test_view_has_card_fields(store):
    a = {**item("a", "https://x.com/1"), "content": "<p>Hello   <b>world</b></p>", "metadata": {"feed_name": "Lobsters"}}
    store.upsert_items([a], "r")
    view = store.list_items(signature="s")[0]
    assert view["feed"] == "Lobsters" and view["excerpt"] == "Hello world" and view["tags"] == []
    store.save_scored("a", tagged(a, ["Rust"]), signature="s")
    view = store.list_items(signature="s")[0]
    assert view["summary"] == "sum" and view["tags"] == ["rust"]


def test_database_without_archived_column_is_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """CREATE TABLE items (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, url_key TEXT NOT NULL,
        source TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL, published_at TEXT, first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL, run_id TEXT, item_json TEXT NOT NULL, scored_json TEXT, score REAL, reason TEXT,
        scored_at TEXT, scored_sig TEXT);
        INSERT INTO items (id,url_key,source,title,url,first_seen,last_seen,item_json)
        VALUES ('a','k','rss','T','https://x.com/1','2026-01-01','2026-01-01','{"metadata":{}}');"""
    )
    conn.commit()
    conn.close()

    migrated = PickStore(path)
    rows = migrated.list_items(signature="s")
    assert len(rows) == 1 and rows[0]["archived"] is False
    PickStore(path)  # opening again is a no-op, not an error


# ---- configuration editor -------------------------------------------------------------------

@pytest.fixture()
def cfg_service(tmp_path, monkeypatch):
    from pathlib import Path

    from src.pick import PickService

    example = Path(__file__).resolve().parents[1] / "data" / "config.example.json"
    config = tmp_path / "config.json"
    config.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    service = PickService(runs_root=tmp_path / "mcp-runs")
    monkeypatch.setattr(service, "config_path", lambda: config)
    return service, config


def test_config_read_returns_file_text(cfg_service):
    service, config = cfg_service
    assert service.read_config_text() == {"path": str(config), "text": config.read_text(encoding="utf-8")}


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("{not json", "Not valid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ('{"ai": {"provider": "nonsense-provider"}}', "rejected by Horizon"),
    ],
)
def test_config_rejects_bad_input_without_touching_the_file(cfg_service, text, fragment):
    service, config = cfg_service
    before = config.read_bytes()
    with pytest.raises(ValueError, match=fragment):
        service.save_config_text(text)
    assert config.read_bytes() == before
    assert not (service.data_dir / "config-backups").exists()


def test_config_save_validates_backs_up_and_replaces_atomically(cfg_service):
    service, config = cfg_service
    original = json.loads(config.read_text(encoding="utf-8"))
    edited = json.loads(json.dumps(original))
    edited["sources"]["rss"].append({"name": "New feed", "url": "https://example.com/feed.xml", "enabled": True})

    result = service.save_config_text(json.dumps(edited))

    saved = json.loads(config.read_text(encoding="utf-8"))
    assert saved["sources"]["rss"][-1]["name"] == "New feed"
    assert result["saved"] is True and result["backup"].startswith("config-")
    backup = service.data_dir / "config-backups" / result["backup"]
    assert json.loads(backup.read_text(encoding="utf-8")) == original  # previous version kept
    assert not list(config.parent.glob(".config.*"))  # no temp files left behind


def test_config_backups_are_pruned(cfg_service, monkeypatch):
    service, config = cfg_service
    monkeypatch.setattr(type(service), "KEEP_BACKUPS", 3)
    backups = service.data_dir / "config-backups"
    backups.mkdir(parents=True)
    for i in range(6):
        (backups / f"config-2026010{i}T000000.json").write_text("{}", encoding="utf-8")
    service.save_config_text(config.read_text(encoding="utf-8"))
    kept = sorted(p.name for p in backups.glob("config-*.json"))
    assert len(kept) == 3 and kept[0] > "config-20260103"  # oldest ones removed


def test_config_too_large_is_rejected(cfg_service):
    service, _ = cfg_service
    with pytest.raises(ValueError, match="too large"):
        service.check_config_text(" " * 1_100_000)


def test_archive_counts(store):
    store.upsert_items([item("a", "https://x.com/1"), item("b", "https://x.com/2"), item("c", "https://x.com/3")], "r")
    assert store.archive_counts() == {"active": 3, "archived": 0}
    store.set_archived([1, 3], True)
    assert store.archive_counts() == {"active": 1, "archived": 2}


# ---- source picker --------------------------------------------------------------------------

def test_source_options_reflect_enabled_sources_in_config(tmp_path):
    from pathlib import Path

    from src.models import Config
    from src.pick import PickService
    from src.storage.manager import _expand_env_vars

    example = json.loads((Path(__file__).resolve().parents[1] / "data" / "config.example.json").read_text(encoding="utf-8"))
    example["sources"]["telegram"]["enabled"] = False
    config = Config.model_validate(_expand_env_vars(example))

    options = {o["id"]: o for o in PickService(runs_root=tmp_path / "r").source_options(config)}
    assert {"hackernews", "rss", "reddit", "github", "telegram"} <= set(options)
    assert options["rss"]["enabled"] and options["rss"]["detail"].endswith("configured")
    assert options["hackernews"]["enabled"]
    assert options["telegram"]["enabled"] is False


@pytest.fixture()
def ui_client():
    from starlette.testclient import TestClient

    from src.pick_ui import app

    return TestClient(app, base_url="http://127.0.0.1:8766")


HEADERS = {"X-Horizon-UI": "1"}


@pytest.mark.parametrize(
    "sources",
    [[], None, "rss", ["rss", "not-a-source"], [1], ["rss; rm -rf /"]],
)
def test_fetch_rejects_empty_or_unknown_sources(ui_client, sources):
    # An empty list would mean "all sources" downstream, so the server must refuse it.
    response = ui_client.post("/api/fetch", json={"hours": 6, "sources": sources}, headers=HEADERS)
    assert response.status_code == 400
    assert response.json()["error"] == "Choose at least one valid source."


def test_fetch_requires_ui_header_and_validates_hours(ui_client):
    assert ui_client.post("/api/fetch", json={"hours": 6, "sources": ["rss"]}).status_code == 403
    response = ui_client.post("/api/fetch", json={"hours": 9999, "sources": ["rss"]}, headers=HEADERS)
    assert response.status_code == 400 and "hours" in response.json()["error"]


def test_tag_variants_collapse_to_one_kebab_case_form(store):
    a, b = item("a", "https://x.com/1"), item("b", "https://x.com/2")
    store.upsert_items([a, b], "r")
    store.save_scored("a", tagged(a, ["Software Engineering"]), signature="s")
    store.save_scored("b", tagged(b, ["#software_engineering", "AI"]), signature="s")
    for variant in ("software-engineering", "Software Engineering", "software_engineering"):
        assert sorted(r["seq"] for r in store.list_items(signature="s", tags=[variant])) == [1, 2]
    assert store.tag_facets(signature="s")[0] == {"tag": "software-engineering", "key": "software-engineering", "count": 2}
