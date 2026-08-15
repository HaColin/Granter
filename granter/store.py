"""The opportunity corpus.

A JSON file written by an ingest run and read by the app. It ships empty: the
repository contains no grant records, because a record that did not come from a
retrieved source document has no business being in the corpus at all.

Run ``python -m granter.ingest`` to populate it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter

from .models import Opportunity, utcnow

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PATH = DATA_DIR / "opportunities.json"

_ADAPTER = TypeAdapter(list[Opportunity])


class Corpus:
    def __init__(self, records: list[Opportunity], fetched_at: datetime | None = None) -> None:
        self.records = records
        self.fetched_at = fetched_at

    def __len__(self) -> int:
        return len(self.records)

    @property
    def is_empty(self) -> bool:
        return not self.records

    def sources(self) -> set[str]:
        return {r.source for r in self.records}


def load(path: Path | None = None) -> Corpus:
    path = path or CORPUS_PATH
    if not path.exists():
        return Corpus([])

    payload = json.loads(path.read_text(encoding="utf-8"))
    records = _ADAPTER.validate_python(payload.get("records", []))
    stamp = payload.get("meta", {}).get("fetched_at")
    return Corpus(records, datetime.fromisoformat(stamp) if stamp else None)


def save(records: list[Opportunity], path: Path | None = None) -> Path:
    path = path or CORPUS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "fetched_at": utcnow().isoformat(),
            "count": len(records),
            "sources": sorted({r.source for r in records}),
        },
        "records": json.loads(_ADAPTER.dump_json(records)),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def merge(existing: list[Opportunity], incoming: list[Opportunity]) -> list[Opportunity]:
    """Incoming records win on id; nothing is silently dropped."""
    by_id = {r.id: r for r in existing}
    by_id.update({r.id: r for r in incoming})
    return sorted(by_id.values(), key=lambda r: (r.source, r.source_id))
