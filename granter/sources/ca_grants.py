"""California Grants Portal connector.

Source: the state's own open-data publication of https://www.grants.ca.gov,
served through CKAN on data.ca.gov and refreshed daily.

    https://data.ca.gov/dataset/california-grants-portal

This is state money, so none of the federal machinery applies: no SAM.gov
registration, no UEI, no Grants.gov account. For a small applicant that is
usually the difference between a call they can realistically enter and one they
cannot.

The same normalising rules hold as everywhere else: a field the state did not
publish becomes ``None`` and is named in ``missing_fields``, and a value that is
published but unreadable is recorded in ``parse_warnings`` rather than dropped.
"""

from __future__ import annotations

import html
import re
from datetime import date, datetime
from typing import Any

import httpx

from ..models import ApplicationStep, Opportunity, utcnow
from ..taxonomy import Sector

CKAN_ENDPOINT = "https://data.ca.gov/api/3/action/datastore_search"
RESOURCE_ID = "111c8c88-21f6-453c-ae2c-b4785a0624f5"

#: The portal's own page for a grant, used when an agency publishes no direct
#: link. It is a state government site, not an aggregator or a middleman.
PORTAL_URL = "https://www.grants.ca.gov/grants/{portal_id}/"

SOURCE = "ca_grants"
PAGE_SIZE = 500

#: The state's applicant vocabulary, mapped onto the Grants.gov codes the
#: eligibility engine already speaks, so both sources grade identically.
#: A public university is a public agency and a private one is a nonprofit
#: institution, which is how the state's own guidance treats them.
APPLICANT_TYPE_CODES: dict[str, frozenset[str]] = {
    "public agency": frozenset({"00", "01", "02", "04", "05", "06", "08"}),
    "nonprofit": frozenset({"12", "13", "20"}),
    "business": frozenset({"22", "23"}),
    "tribal government": frozenset({"07", "11"}),
    "individual": frozenset({"21"}),
    # Deliberately mapped to the "Others" code so it reads as "check the text"
    # rather than as a claim about who qualifies.
    "other legal entity": frozenset({"25"}),
}

CATEGORY_SECTORS: dict[str, tuple[Sector, ...]] = {
    "environment & water": (Sector.ENVIRONMENT,),
    "energy": (Sector.ENVIRONMENT,),
    "parks & recreation": (Sector.ENVIRONMENT,),
    "agriculture": (Sector.FOOD_AGRICULTURE,),
    "food & nutrition": (Sector.FOOD_AGRICULTURE,),
    "education": (Sector.EDUCATION,),
    "health & human services": (Sector.HEALTH,),
    "housing, community and economic development": (
        Sector.HOUSING,
        Sector.COMMUNITY_DEVELOPMENT,
    ),
    "disadvantaged communities": (Sector.COMMUNITY_DEVELOPMENT,),
    "employment, labor & training": (Sector.COMMUNITY_DEVELOPMENT,),
    "science, technology, and research & development": (
        Sector.TECHNOLOGY,
        Sector.RESEARCH,
    ),
    "transportation": (Sector.INFRASTRUCTURE,),
    "disaster prevention & relief": (Sector.INFRASTRUCTURE,),
    "law, justice, and legal services": (Sector.HUMAN_RIGHTS,),
    "libraries and arts": (Sector.ARTS,),
}


class SourceShapeError(RuntimeError):
    """The portal returned something this normaliser does not understand."""


# --- HTTP -------------------------------------------------------------------


def fetch_rows(
    client: httpx.Client,
    *,
    status: str = "active",
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Page through the datastore, newest first, filtered by status."""
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < limit:
        response = client.get(
            CKAN_ENDPOINT,
            params={
                "resource_id": RESOURCE_ID,
                "limit": min(PAGE_SIZE, limit - len(rows)),
                "offset": offset,
                "filters": f'{{"Status":"{status}"}}',
            },
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise SourceShapeError(f"datastore_search failed: {payload.get('error')}")

        result = payload.get("result") or {}
        page = result.get("records")
        if not isinstance(page, list):
            raise SourceShapeError(f"no 'records' list; result keys were {sorted(result)}")

        rows.extend(page)
        offset += len(page)
        if not page or offset >= int(result.get("total") or 0):
            break
    return rows


# --- Normalisation ----------------------------------------------------------


def _clean(value: Any) -> str:
    if value in (None, "None", "N/A", "null"):
        return ""
    return re.sub(r"\s+", " ", html.unescape(str(value))).strip()


def _parse_date(value: Any, warnings: list[str], field: str) -> date | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    warnings.append(f"{field}={text!r} did not match any known format")
    return None


#: How the portal expresses "no fixed deadline". These are not parse failures:
#: a rolling call is a fact about the grant, and for a small applicant it is
#: often the most useful fact there is.
ROLLING_MARKERS = ("ongoing", "rolling", "continuous", "open until filled", "varies")


def _parse_deadline(value: Any, warnings: list[str]) -> tuple[date | None, bool]:
    """Return (close date, is rolling)."""
    text = _clean(value)
    if not text:
        return None, False
    if text.lower() in ROLLING_MARKERS:
        return None, True
    return _parse_date(text, warnings, "close_date"), False


_MONEY = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


def _parse_amounts(value: Any) -> tuple[int | None, int | None]:
    """Read the state's free-text award range.

    Published as "Between $50,000 and $500,000", or a single "$500,000", or
    prose like "Dependant on number of submissions received". Only the first two
    carry a number; the prose is not guessed at.

    A single amount sets the ceiling and leaves the floor unknown -- it says how
    large an award can be, not that every award is exactly that size.
    """
    text = _clean(value)
    if not text:
        return None, None

    amounts = [int(float(m.replace(",", ""))) for m in _MONEY.findall(text)]
    amounts = [a for a in amounts if a > 0]

    if len(amounts) >= 2:
        return min(amounts), max(amounts)
    if len(amounts) == 1:
        return None, amounts[0]
    return None, None


def _applicant_codes(value: Any) -> list[str]:
    codes: set[str] = set()
    for part in _clean(value).split(";"):
        codes |= APPLICANT_TYPE_CODES.get(part.strip().lower(), frozenset())
    return sorted(codes)


def _sectors(value: Any) -> list[Sector]:
    found: set[Sector] = set()
    for part in _clean(value).split(";"):
        found.update(CATEGORY_SECTORS.get(part.strip().lower(), ()))
    return sorted(found, key=lambda s: s.value)


def _cost_share(value: Any) -> bool | None:
    text = _clean(value).lower()
    if not text:
        return None
    if text in ("not required", "none", "0%"):
        return False
    return True


def _steps(url: str, contact: str) -> list[ApplicationStep]:
    """State grants have no central submission system to describe generically.

    Rather than invent a process, point at the agency's own page and its
    published contact, and say plainly that the agency defines the rest.
    """
    steps = [
        ApplicationStep(
            order=1,
            description="Read the grant page on the administering agency's own site.",
            url=url,
        ),
        ApplicationStep(
            order=2,
            description=(
                "Confirm eligibility and the submission method with the agency — California "
                "agencies each run their own process, and there is no single state portal "
                "that accepts applications."
            ),
        ),
    ]
    if contact:
        steps.append(
            ApplicationStep(order=3, description=f"Agency contact: {contact}")
        )
    return steps


def normalise(row: dict[str, Any]) -> Opportunity:
    """Turn one portal row into an Opportunity record."""
    portal_id = _clean(row.get("PortalID"))
    if not portal_id:
        raise SourceShapeError(f"row has no PortalID; keys were {sorted(row)}")

    title = _clean(row.get("Title"))
    if not title:
        raise SourceShapeError(f"grant {portal_id} has no Title")

    missing: list[str] = []
    warnings: list[str] = []

    def track(name: str, value: Any) -> Any:
        if value in (None, "", []):
            missing.append(name)
        return value

    close_date, rolling = _parse_deadline(row.get("ApplicationDeadline"), warnings)
    if close_date is None and not rolling:
        missing.append("close_date")
    floor, ceiling = _parse_amounts(row.get("EstAmounts"))
    if floor is None:
        missing.append("award_floor")
    if ceiling is None:
        missing.append("award_ceiling")

    codes = track("applicant_codes", _applicant_codes(row.get("ApplicantType")))
    cost_share = _cost_share(row.get("MatchingFunds"))
    if cost_share is None:
        missing.append("cost_share_required")

    url = _clean(row.get("GrantURL")) or PORTAL_URL.format(portal_id=portal_id)
    contact = _clean(row.get("ContactInfo"))

    # Eligibility notes the state publishes separately from the applicant list.
    eligibility = " ".join(
        part
        for part in (_clean(row.get("ApplicantTypeNotes")), _clean(row.get("Geography")))
        if part
    )

    _, pool = _parse_amounts(row.get("EstAvailFunds"))

    return Opportunity(
        id=f"{SOURCE}:{portal_id}",
        source=SOURCE,
        source_id=_clean(row.get("GrantID")) or portal_id,
        title=title,
        funder=_clean(row.get("AgencyDept")) or "State of California",
        source_url=url,
        description=" ".join(
            part for part in (_clean(row.get("Purpose")), _clean(row.get("Description"))) if part
        ),
        eligibility_text=eligibility,
        applicant_codes=codes,
        award_floor=floor,
        award_ceiling=ceiling,
        total_pool=pool,
        posted_date=_parse_date(row.get("OpenDate"), warnings, "posted_date"),
        close_date=close_date,
        rolling=rolling,
        is_forecast=_clean(row.get("Status")).lower() == "forecasted",
        jurisdiction="US",
        region="CA",
        eligible_work_countries=["US"],
        sectors=_sectors(row.get("Categories")),
        cost_share_required=cost_share,
        # No SAM.gov, no UEI, no Grants.gov account. That is the point of this
        # source for a small applicant, so the absence is left genuinely empty.
        prerequisites=[],
        forms=[],
        steps=_steps(url, contact),
        fetched_at=utcnow(),
        missing_fields=sorted(set(missing)),
        parse_warnings=warnings,
    )


def collect(
    limit: int = 1000,
    status: str = "active",
    client: httpx.Client | None = None,
) -> list[Opportunity]:
    owned = client is None
    client = client or httpx.Client(
        headers={"User-Agent": "Granter/0.1 (grant discovery)"}, follow_redirects=True
    )
    records: list[Opportunity] = []
    try:
        for row in fetch_rows(client, status=status, limit=limit):
            try:
                records.append(normalise(row))
            except SourceShapeError as exc:
                print(f"  skipped CA row {row.get('PortalID')}: {exc}")
    finally:
        if owned:
            client.close()
    return records
