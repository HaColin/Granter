"""Populate the opportunity corpus from live sources.

    python -m granter.ingest                 # all open Grants.gov calls, up to --limit
    python -m granter.ingest --keyword water --limit 200
    python -m granter.ingest --probe         # verify API reachability and payload shape

Nothing here invents records. If a source is unreachable the run fails and the
existing corpus is left untouched.
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

from . import store
from .sources import grants_gov


RAW_DUMP = store.DATA_DIR / "probe-payload.json"


def probe(statuses: str = "posted") -> int:
    """Fetch one opportunity, report the payload shape, and try to normalise it.

    Diagnostics are printed as they are gathered, so a normalisation failure
    still leaves you with the field names needed to fix it. The raw payload is
    written to ``data/probe-payload.json`` for exactly that purpose.
    """
    try:
        with httpx.Client(headers={"User-Agent": "Granter/0.1 (probe)"}) as client:
            hits = grants_gov.search(client, statuses=statuses, limit=1)
            if not hits:
                print(f"search2 returned no hits for statuses={statuses!r}", file=sys.stderr)
                return 1

            print(f"search2 hit keys:       {sorted(hits[0])}")
            detail = grants_gov.fetch_detail(client, hits[0]["id"])
            print(f"fetchOpportunity keys:  {sorted(detail)}")

            RAW_DUMP.parent.mkdir(parents=True, exist_ok=True)
            RAW_DUMP.write_text(json.dumps(detail, indent=2), encoding="utf-8")
            print(f"raw payload written to: {RAW_DUMP}")

            for name in grants_gov.DETAIL_BLOCKS:
                block = detail.get(name)
                if isinstance(block, dict):
                    print(f"{name} keys: {sorted(block)}")
    except (httpx.HTTPError, OSError) as exc:
        print(f"probe failed before it could read the payload: {exc}", file=sys.stderr)
        return 1

    try:
        record = grants_gov.normalise(detail)
    except grants_gov.SourceShapeError as exc:
        print(f"\nnormalise failed: {exc}", file=sys.stderr)
        print("The keys above are what the API actually returned. Send them along "
              f"with {RAW_DUMP.name} to fix the normaliser.", file=sys.stderr)
        return 1

    print("\nnormalised record:")
    print(json.dumps(record.model_dump(mode="json"), indent=2, default=str)[:2000])
    if record.missing_fields:
        print(f"\nfields the source did not publish: {record.missing_fields}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch grant opportunities into the local corpus.")
    parser.add_argument("--keyword", default="", help="restrict the search to a keyword")
    parser.add_argument("--limit", type=int, default=100, help="maximum opportunities to fetch")
    parser.add_argument("--replace", action="store_true", help="discard the existing corpus")
    parser.add_argument("--probe", action="store_true", help="check API shape and exit")
    parser.add_argument(
        "--include-forecasted",
        action="store_true",
        help="also fetch forecasts -- announced intentions to fund, not yet open to apply",
    )
    args = parser.parse_args(argv)

    statuses = "posted|forecasted" if args.include_forecasted else "posted"

    if args.probe:
        return probe(statuses)

    print(f"Fetching up to {args.limit} opportunities from Grants.gov (statuses={statuses})...")
    try:
        records = grants_gov.collect(keyword=args.keyword, limit=args.limit, statuses=statuses)
    except (httpx.HTTPError, grants_gov.SourceShapeError) as exc:
        print(f"ingest failed, corpus unchanged: {exc}", file=sys.stderr)
        return 1

    if not records:
        print("no usable records returned; corpus unchanged", file=sys.stderr)
        return 1

    existing = [] if args.replace else store.load().records
    merged = store.merge(existing, records)
    path = store.save(merged)

    incomplete = sum(1 for r in merged if r.missing_fields)
    print(f"wrote {len(merged)} records to {path}")
    print(f"  {len(records)} fetched this run, {incomplete} with fields the source did not publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
