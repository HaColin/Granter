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


def probe() -> int:
    """Fetch one opportunity and print the raw payload keys.

    Use this after any upstream API change to confirm the field names the
    normaliser depends on still exist.
    """
    try:
        with httpx.Client(headers={"User-Agent": "Granter/0.1 (probe)"}) as client:
            hits = grants_gov.search(client, limit=1)
            if not hits:
                print("search2 returned no hits", file=sys.stderr)
                return 1
            print(f"search2 hit keys: {sorted(hits[0])}")
            detail = grants_gov.fetch_detail(client, hits[0]["id"])
            print(f"fetchOpportunity keys: {sorted(detail)}")
            synopsis = detail.get("synopsis") or {}
            print(f"synopsis keys: {sorted(synopsis)}")
            print(json.dumps(grants_gov.normalise(detail).model_dump(mode="json"), indent=2)[:2000])
    except (httpx.HTTPError, grants_gov.SourceShapeError) as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch grant opportunities into the local corpus.")
    parser.add_argument("--keyword", default="", help="restrict the search to a keyword")
    parser.add_argument("--limit", type=int, default=100, help="maximum opportunities to fetch")
    parser.add_argument("--replace", action="store_true", help="discard the existing corpus")
    parser.add_argument("--probe", action="store_true", help="check API shape and exit")
    args = parser.parse_args(argv)

    if args.probe:
        return probe()

    print(f"Fetching up to {args.limit} opportunities from Grants.gov...")
    try:
        records = grants_gov.collect(keyword=args.keyword, limit=args.limit)
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
