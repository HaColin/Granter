"""Connector HTTP behaviour, driven through a mock transport.

This covers paging, error propagation, and the skip-rather-than-patch policy
without touching the network. What it deliberately cannot prove is that the live
Grants.gov API still uses these field names -- that is what
``python -m granter.ingest --probe`` is for.
"""

from __future__ import annotations

import httpx
import pytest

from granter.sources import grants_gov


def synopsis(**overrides) -> dict:
    base = {
        "agencyName": "Test Agency",
        "responseDate": "12312026",
        "awardCeiling": "150000",
        "awardFloor": "25000",
        "applicantTypes": [{"id": "12"}],
        "synopsisDesc": "Description.",
        "costSharing": False,
    }
    base.update(overrides)
    return base


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def ok(data: dict) -> httpx.Response:
    return httpx.Response(200, json={"errorcode": 0, "msg": "success", "data": data})


def test_search_pages_until_the_hit_count_is_exhausted():
    seen_starts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        seen_starts.append(body["startRecordNum"])
        start = body["startRecordNum"]
        page = [{"id": i} for i in range(start, min(start + grants_gov.PAGE_SIZE, 150))]
        return ok({"hitCount": 150, "oppHits": page})

    with make_client(handler) as client:
        hits = grants_gov.search(client, limit=150)

    assert len(hits) == 150
    assert seen_starts == [0, 100]


def test_search_stops_at_the_requested_limit():
    def handler(request: httpx.Request) -> httpx.Response:
        return ok({"hitCount": 1000, "oppHits": [{"id": i} for i in range(10)]})

    with make_client(handler) as client:
        assert len(grants_gov.search(client, limit=10)) == 10


def test_api_error_code_becomes_a_source_shape_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"errorcode": 1, "msg": "bad request"})

    with make_client(handler) as client, pytest.raises(grants_gov.SourceShapeError, match="errorcode"):
        grants_gov.search(client)


def test_missing_data_object_becomes_a_source_shape_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"errorcode": 0, "msg": "success"})

    with make_client(handler) as client, pytest.raises(grants_gov.SourceShapeError, match="no 'data'"):
        grants_gov.search(client)


def test_http_error_propagates_rather_than_returning_an_empty_page():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with make_client(handler) as client, pytest.raises(httpx.HTTPStatusError):
        grants_gov.search(client)


def test_collect_normalises_every_usable_hit():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("search2"):
            return ok({"hitCount": 2, "oppHits": [{"id": 1}, {"id": 2}]})
        import json

        opp_id = json.loads(request.content)["opportunityId"]
        return ok({
            "id": opp_id,
            "opportunityNumber": f"TEST-{opp_id}",
            "opportunityTitle": f"Call {opp_id}",
            "synopsis": synopsis(),
        })

    with make_client(handler) as client:
        records = grants_gov.collect(limit=2, client=client)

    assert [r.source_id for r in records] == ["TEST-1", "TEST-2"]
    assert all(str(r.source_url).startswith("https://www.grants.gov/") for r in records)
    assert all(r.fetched_at is not None for r in records)


def test_collect_skips_an_unnormalisable_record_instead_of_patching_it(capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("search2"):
            return ok({"hitCount": 2, "oppHits": [{"id": 1}, {"id": 2}]})
        import json

        opp_id = json.loads(request.content)["opportunityId"]
        if opp_id == 1:
            return ok({"id": 1})  # no synopsis
        return ok({
            "id": 2,
            "opportunityNumber": "TEST-2",
            "opportunityTitle": "Call 2",
            "synopsis": synopsis(),
        })

    with make_client(handler) as client:
        records = grants_gov.collect(limit=2, client=client)

    assert [r.source_id for r in records] == ["TEST-2"]
    assert "skipped opportunity 1" in capsys.readouterr().out


def test_hits_without_an_id_are_ignored():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("search2"):
            return ok({"hitCount": 1, "oppHits": [{"number": "no-id-here"}]})
        raise AssertionError("should not fetch details for a hit with no id")

    with make_client(handler) as client:
        assert grants_gov.collect(limit=1, client=client) == []


# --- date parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("12312026", "2026-12-31"),
        ("2026-12-31", "2026-12-31"),
        ("12/31/2026", "2026-12-31"),
        ("2026/12/31", "2026-12-31"),
        ("31-Dec-2026", "2026-12-31"),
        ("Dec 31, 2026", "2026-12-31"),
        ("December 31, 2026", "2026-12-31"),
        ("2026-12-31T00:00:00", "2026-12-31"),
        ("2026-12-31T00:00:00Z", "2026-12-31"),
        ("2026-12-31T00:00:00.000+00:00", "2026-12-31"),
        # Real values observed in a live Grants.gov ingest: a date carrying a
        # time and a named US timezone, which strptime cannot read portably.
        ("Aug 07, 2028 12:00:00 AM EDT", "2028-08-07"),
        ("Dec 31, 2026 11:59:59 PM EST", "2026-12-31"),
        ("December 31, 2026 05:00:00 PM PST", "2026-12-31"),
        ("12/31/2026 12:00:00 AM EDT", "2026-12-31"),
    ],
)
def test_every_plausible_date_shape_is_read(raw, expected):
    assert grants_gov._parse_date(raw).isoformat() == expected


def test_an_absent_date_produces_no_warning():
    warnings: list[str] = []
    assert grants_gov._parse_date(None, warnings, "close_date") is None
    assert grants_gov._parse_date("", warnings, "close_date") is None
    assert warnings == []


def test_an_unreadable_date_is_reported_not_swallowed():
    """A published value we cannot read is a bug here, not a gap in the source."""
    warnings: list[str] = []
    assert grants_gov._parse_date("sometime next spring", warnings, "close_date") is None
    assert warnings and "close_date" in warnings[0] and "next spring" in warnings[0]


def test_parse_warnings_reach_the_record():
    record = grants_gov.normalise({
        "id": 5,
        "opportunityTitle": "T",
        "synopsis": {**synopsis(), "responseDate": "whenever"},
    })
    assert record.close_date is None
    assert record.parse_warnings and "close_date" in record.parse_warnings[0]
