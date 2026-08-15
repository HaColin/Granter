"""EU Funding & Tenders Portal connector.

Source: the portal's own bulk reference file, which is the complete set of
calls and tenders it publishes.

    https://ec.europa.eu/info/funding-tenders/opportunities/data/referenceData/grantsTenders.json

Two things about this source shape the connector.

It is a single ~128 MB document, so it is cached on disk and only re-downloaded
when the cache is stale. And it publishes no award amounts and no descriptive
text beyond titles and tags -- so every record here has ``award_floor`` and
``award_ceiling`` as ``None`` with those names in ``missing_fields``, and its
relevance score is computed from a thinner text than a Grants.gov record.
That is a real limitation of the source, and it is recorded rather than papered
over with an invented budget.

Tenders (``type`` 0) are procurement contracts, not grants, and are excluded.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from ..models import ApplicationStep, Opportunity, utcnow
from ..taxonomy import Sector

BULK_URL = (
    "https://ec.europa.eu/info/funding-tenders/opportunities/data/referenceData/"
    "grantsTenders.json"
)
TOPIC_URL = (
    "https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/"
    "opportunities/topic-details/{identifier}"
)

SOURCE = "eu_portal"

#: ``type`` 1 is a call for proposals (a grant); 0 is a tender (a contract).
GRANT_TYPE = "1"

OPEN_STATUSES = {"Open"}
FORTHCOMING_STATUSES = {"Forthcoming"}

#: Refetch the bulk file at most once a day; it is large and changes slowly.
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60

#: EU calls are open to legal entities, normally established in a Member State
#: or associated country, and often require a consortium. The portal publishes
#: no structured applicant-type field, so this is stated as text rather than
#: encoded as eligibility codes that were never in the source.
ELIGIBILITY_PREAMBLE = (
    "The portal publishes no structured eligible-applicant list for this call. EU "
    "programmes are generally open to legal entities established in an EU Member State "
    "or associated country, frequently require a consortium of partners from several "
    "countries, and are rarely open to individuals. Read the call's eligibility "
    "conditions on the topic page before investing time."
)

PROGRAMME_SECTORS: dict[str, tuple[Sector, ...]] = {
    "horizon": (Sector.RESEARCH,),
    "erasmus": (Sector.EDUCATION,),
    "life": (Sector.ENVIRONMENT,),
    "cef": (Sector.INFRASTRUCTURE,),
    "digital": (Sector.TECHNOLOGY,),
    "eu4health": (Sector.HEALTH,),
    "creative": (Sector.ARTS,),
    "cerv": (Sector.HUMAN_RIGHTS,),
    "just": (Sector.HUMAN_RIGHTS,),
    "amif": (Sector.HUMAN_RIGHTS,),
    "agrip": (Sector.FOOD_AGRICULTURE,),
    "esf": (Sector.COMMUNITY_DEVELOPMENT,),
    "smp": (Sector.COMMUNITY_DEVELOPMENT,),
}


class SourceShapeError(RuntimeError):
    """The portal returned something this normaliser does not understand."""


# --- Fetching ---------------------------------------------------------------


def _cache_path() -> Path:
    from .. import store

    return store.DATA_DIR / "eu_grants_tenders_raw.json"


def fetch_bulk(client: httpx.Client | None = None, max_age: int = CACHE_MAX_AGE_SECONDS) -> dict:
    """Download the reference file, or reuse a recent copy from disk."""
    cache = _cache_path()
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age:
        return json.loads(cache.read_text(encoding="utf-8"))

    owned = client is None
    client = client or httpx.Client(
        headers={"User-Agent": "Granter/0.1 (grant discovery)"}, follow_redirects=True
    )
    try:
        response = client.get(BULK_URL, timeout=300.0)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owned:
            client.close()

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def extract_objects(payload: dict) -> list[dict[str, Any]]:
    objects = (payload.get("fundingData") or {}).get("GrantTenderObj")
    if not isinstance(objects, list):
        raise SourceShapeError(
            f"no fundingData.GrantTenderObj list; top-level keys were {sorted(payload)}"
        )
    return objects


# --- Normalisation ----------------------------------------------------------


def _status(row: dict[str, Any]) -> str:
    status = row.get("status")
    if isinstance(status, dict):
        return str(status.get("abbreviation") or "")
    return str(status or "")


def _epoch_date(value: Any) -> date | None:
    """Dates arrive as epoch milliseconds."""
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).date()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _next_deadline(row: dict[str, Any], today: date) -> tuple[date | None, int]:
    """The next cut-off, and how many remain.

    A call can publish several cut-off dates. The one that matters is the next
    one that has not passed; showing the first in the list would report a
    deadline that expired years ago.
    """
    dates = [d for d in (_epoch_date(v) for v in row.get("deadlineDatesLong") or []) if d]
    if not dates:
        return None, 0
    upcoming = sorted(d for d in dates if d >= today)
    if upcoming:
        return upcoming[0], len(upcoming)
    return max(dates), 0


def _programme(row: dict[str, Any]) -> str:
    framework = row.get("frameworkProgramme")
    if isinstance(framework, dict):
        return str(framework.get("description") or framework.get("abbreviation") or "").strip()
    return ""


def _sectors(row: dict[str, Any], programme: str) -> list[Sector]:
    haystack = f"{programme} {row.get('callIdentifier', '')} {row.get('identifier', '')}".lower()
    found: set[Sector] = set()
    for marker, sectors in PROGRAMME_SECTORS.items():
        if marker in haystack:
            found.update(sectors)
    return sorted(found, key=lambda s: s.value)


def normalise(row: dict[str, Any], today: date | None = None) -> Opportunity:
    today = today or date.today()

    identifier = str(row.get("identifier") or "").strip()
    if not identifier:
        raise SourceShapeError(f"row has no identifier; keys were {sorted(row)}")

    title = str(row.get("title") or row.get("callTitle") or "").strip()
    if not title:
        raise SourceShapeError(f"call {identifier} has no title")

    missing: list[str] = []
    programme = _programme(row)
    close_date, remaining_cutoffs = _next_deadline(row, today)
    if close_date is None:
        missing.append("close_date")

    # The source carries no descriptive prose, so the searchable text is built
    # from the fields that do exist. Thin, but not invented.
    tags = [str(t) for t in (row.get("tags") or []) if t]
    keywords = [str(k) for k in (row.get("keywords") or []) if k]
    call_title = str(row.get("callTitle") or "").strip()
    description = " ".join(
        part for part in [call_title, programme, " ".join(tags), " ".join(keywords)] if part
    )

    eligibility = ELIGIBILITY_PREAMBLE
    if row.get("sme"):
        eligibility += " The portal flags this call as relevant to SMEs."
    if remaining_cutoffs > 1:
        eligibility += f" This call has {remaining_cutoffs} further cut-off dates."

    # No award amounts exist anywhere in this dataset.
    missing += ["award_ceiling", "award_floor"]

    url = TOPIC_URL.format(identifier=identifier)

    return Opportunity(
        id=f"{SOURCE}:{identifier}",
        source=SOURCE,
        source_id=str(row.get("callIdentifier") or identifier),
        title=title,
        funder=f"European Commission — {programme}" if programme else "European Commission",
        source_url=url,
        description=description,
        eligibility_text=eligibility,
        # Deliberately empty: the source states no applicant types, and the
        # engine reports an empty list as "not published" rather than a pass.
        applicant_codes=[],
        posted_date=_epoch_date(row.get("publicationDateLong")),
        close_date=close_date,
        is_forecast=_status(row) in FORTHCOMING_STATUSES,
        jurisdiction="EU",
        sectors=_sectors(row, programme),
        prerequisites=[
            "EU Login account",
            "Participant Identification Code (PIC) via the Participant Register",
        ],
        forms=[],
        steps=[
            ApplicationStep(
                order=1,
                description="Read the topic page, its call document and eligibility conditions.",
                url=url,
            ),
            ApplicationStep(
                order=2,
                description="Create an EU Login account and register your organisation for a PIC.",
                url="https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/how-to-participate/participant-register",
                lead_time_days=14,
            ),
            ApplicationStep(
                order=3,
                description=(
                    "Check whether the call requires a consortium — many EU calls need partners "
                    "from several countries — and use the portal's partner search if so."
                ),
                url=url,
            ),
            ApplicationStep(
                order=4,
                description="Submit through the portal's electronic submission system before the cut-off.",
                url=url,
            ),
        ],
        fetched_at=utcnow(),
        missing_fields=sorted(set(missing)),
    )


def collect(
    limit: int = 2000,
    include_forthcoming: bool = False,
    client: httpx.Client | None = None,
    today: date | None = None,
) -> list[Opportunity]:
    today = today or date.today()
    wanted = OPEN_STATUSES | (FORTHCOMING_STATUSES if include_forthcoming else set())

    rows = [
        row
        for row in extract_objects(fetch_bulk(client))
        if str(row.get("type")) == GRANT_TYPE and _status(row) in wanted
    ]

    records: list[Opportunity] = []
    for row in rows[:limit]:
        try:
            records.append(normalise(row, today=today))
        except SourceShapeError as exc:
            print(f"  skipped EU row {row.get('identifier')}: {exc}")
    return records
