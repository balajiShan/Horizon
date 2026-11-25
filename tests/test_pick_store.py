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
