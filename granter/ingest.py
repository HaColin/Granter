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
from .sources import ca_grants, eu_portal, grants_gov


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
        "--source",
        choices=("all", "grants_gov", "ca_grants", "eu_portal"),
        default="all",
        help="which source(s) to fetch from (default: all)",
    )
    parser.add_argument(
        "--include-forecasted",
        action="store_true",
        help="also fetch forecasts -- announced intentions to fund, not yet open to apply",
    )
    args = parser.parse_args(argv)

    statuses = "posted|forecasted" if args.include_forecasted else "posted"

    if args.probe:
        return probe(statuses)

    records: list = []
    failures: list[str] = []

    if args.source in ("all", "grants_gov"):
        print(f"Fetching up to {args.limit} from Grants.gov (federal, statuses={statuses})...")
        try:
            fetched = grants_gov.collect(keyword=args.keyword, limit=args.limit, statuses=statuses)
            print(f"  {len(fetched)} records")
            records += fetched
        except (httpx.HTTPError, grants_gov.SourceShapeError) as exc:
            failures.append(f"grants_gov: {exc}")

    if args.source in ("all", "ca_grants"):
        print(f"Fetching up to {args.limit} from the California Grants Portal (state)...")
        try:
            fetched = ca_grants.collect(limit=args.limit)
            if args.include_forecasted:
                fetched += ca_grants.collect(limit=args.limit, status="forecasted")
            if args.keyword:
                # The portal has no keyword filter, so it is applied here.
                needle = args.keyword.lower()
                fetched = [
                    r for r in fetched
                    if needle in f"{r.title} {r.description} {r.eligibility_text}".lower()
                ]
            print(f"  {len(fetched)} records")
            records += fetched
        except (httpx.HTTPError, ca_grants.SourceShapeError) as exc:
            failures.append(f"ca_grants: {exc}")

    if args.source in ("all", "eu_portal"):
        print("Fetching from the EU Funding & Tenders Portal (bulk file, cached daily)...")
        try:
            fetched = eu_portal.collect(
                limit=args.limit, include_forthcoming=args.include_forecasted
            )
            if args.keyword:
                needle = args.keyword.lower()
                fetched = [
                    r for r in fetched
                    if needle in f"{r.title} {r.description} {r.eligibility_text}".lower()
                ]
            print(f"  {len(fetched)} records")
            records += fetched
        except (httpx.HTTPError, eu_portal.SourceShapeError, OSError) as exc:
            failures.append(f"eu_portal: {exc}")

    for failure in failures:
        print(f"source failed: {failure}", file=sys.stderr)

    if not records:
        print("no usable records returned; corpus unchanged", file=sys.stderr)
        return 1

    # --replace discards what this run refetched, not what it failed to reach.
    # A source that is down must not silently delete its records: the corpus
    # would shrink without anyone asking for that, and the UI would report
    # "no matches" for a source that simply had a bad minute.
    refreshed = {r.source for r in records}
    if args.replace:
        existing = [r for r in store.load().records if r.source not in refreshed]
        if existing:
            kept = ", ".join(sorted({r.source for r in existing}))
            print(f"  --replace kept {len(existing)} record(s) from sources not fetched: {kept}")
    else:
        existing = store.load().records

    merged = store.merge(existing, records)
    path = store.save(merged)

    from collections import Counter

    incomplete = sum(1 for r in merged if r.missing_fields)
    print(f"\nwrote {len(merged)} records to {path}")
    for source, count in sorted(Counter(r.source for r in merged).items()):
        print(f"  {source}: {count}")
    print(f"  {len(records)} fetched this run, {incomplete} with fields the source did not publish")

    with_dates = sum(1 for r in records if r.close_date or r.rolling)
    print(f"  {with_dates}/{len(records)} have a usable deadline")

    # A value the source published but this code could not read is a bug here.
    # Print it rather than letting a whole corpus quietly lose its deadlines.
    unreadable = [w for r in records for w in r.parse_warnings]
    if unreadable:
        print(f"\n  {len(unreadable)} value(s) could not be parsed -- this is a bug in the "
              "normaliser, not missing source data:", file=sys.stderr)
        for warning in sorted(set(unreadable))[:10]:
            print(f"    {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
