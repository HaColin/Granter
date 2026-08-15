"""Grants.gov connector.

Uses the public Search2 / FetchOpportunity JSON API. No key is required at the
time of writing; check https://www.grants.gov/api/api-guide for current terms
before running this at volume.

The normaliser is deliberately strict. Where the payload does not contain a
field, the record gets ``None`` and the name is appended to ``missing_fields``
so the UI can say "not published" instead of implying a value. If the payload
shape changes, :func:`normalise` raises rather than emitting a partial record
that would look authoritative in the results list.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import httpx

from ..models import ApplicationStep, FormReference, Opportunity, utcnow

BASE_URL = "https://api.grants.gov/v1/api"
SEARCH_ENDPOINT = f"{BASE_URL}/search2"
FETCH_ENDPOINT = f"{BASE_URL}/fetchOpportunity"

#: The public detail page for an opportunity -- the official link users get.
DETAIL_URL = "https://www.grants.gov/search-results-detail/{id}"

SOURCE = "grants_gov"
PAGE_SIZE = 100


class SourceShapeError(RuntimeError):
    """The API returned something this normaliser does not understand."""


# --- HTTP -------------------------------------------------------------------


def _post(client: httpx.Client, url: str, body: dict[str, Any]) -> dict[str, Any]:
    response = client.post(url, json=body, timeout=30.0)
    response.raise_for_status()
    payload = response.json()
    if payload.get("errorcode") not in (0, None):
        raise SourceShapeError(f"{url} returned errorcode={payload.get('errorcode')}: {payload.get('msg')}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise SourceShapeError(f"{url} returned no 'data' object; got keys {sorted(payload)}")
    return data


def search(
    client: httpx.Client,
    *,
    keyword: str = "",
    statuses: str = "posted|forecasted",
    limit: int = PAGE_SIZE,
) -> list[dict[str, Any]]:
    """Return raw search hits (id + summary fields), paging as needed."""
    hits: list[dict[str, Any]] = []
    start = 0
    while len(hits) < limit:
        data = _post(
            client,
            SEARCH_ENDPOINT,
            {
                "keyword": keyword,
                "oppStatuses": statuses,
                "rows": min(PAGE_SIZE, limit - len(hits)),
                "startRecordNum": start,
            },
        )
        page = data.get("oppHits") or []
        if not isinstance(page, list):
            raise SourceShapeError("search2 'oppHits' was not a list")
        hits.extend(page)
        start += len(page)
        if len(page) < PAGE_SIZE or start >= int(data.get("hitCount") or 0):
            break
    return hits


def fetch_detail(client: httpx.Client, opportunity_id: str | int) -> dict[str, Any]:
    return _post(client, FETCH_ENDPOINT, {"opportunityId": int(opportunity_id)})


# --- Normalisation ----------------------------------------------------------


def _parse_date(value: Any) -> date | None:
    """Grants.gov publishes dates as MMDDYYYY, occasionally as ISO."""
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%m%d%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _parse_money(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        return int(float(str(value).replace(",", "").replace("$", "")))
    except ValueError:
        return None


def _applicant_codes(synopsis: dict[str, Any]) -> list[str]:
    raw = synopsis.get("applicantTypes") or []
    codes: list[str] = []
    for entry in raw:
        if isinstance(entry, dict):
            code = entry.get("id") or entry.get("code")
            if code is not None:
                codes.append(str(code).zfill(2))
        elif isinstance(entry, str):
            codes.append(entry.zfill(2))
    return codes


def _forms(detail: dict[str, Any]) -> list[FormReference]:
    """Collect application packages. Emits nothing when the payload has none.

    The UI renders "No Forms Found" for an empty list rather than guessing at a
    package URL, because a wrong form link costs the applicant more than a
    missing one.
    """
    forms: list[FormReference] = []
    for pkg in detail.get("opportunityPkgs") or []:
        if not isinstance(pkg, dict):
            continue
        name = (
            pkg.get("competitionTitle")
            or pkg.get("packageId")
            or pkg.get("dialect")
            or "Application package"
        )
        url = pkg.get("instructionsUrl") or pkg.get("url")
        forms.append(FormReference(name=str(name), url=url if url else None, required=True))
    return forms


def _steps(opportunity_id: str, forms: list[FormReference]) -> list[ApplicationStep]:
    """The fixed federal submission path, in order."""
    steps = [
        ApplicationStep(
            order=1,
            description="Obtain a Unique Entity ID and register your organisation in SAM.gov.",
            url="https://sam.gov/content/entity-registration",
            lead_time_days=30,
        ),
        ApplicationStep(
            order=2,
            description="Create a Grants.gov account and get the Authorized Organization Representative role.",
            url="https://www.grants.gov/applicants/registration",
            lead_time_days=7,
        ),
        ApplicationStep(
            order=3,
            description="Read the full funding opportunity announcement, including the eligibility section.",
            url=DETAIL_URL.format(id=opportunity_id),
        ),
    ]
    if forms:
        steps.append(
            ApplicationStep(
                order=4,
                description="Download and complete the application package forms listed below.",
                url=DETAIL_URL.format(id=opportunity_id),
            )
        )
    steps.append(
        ApplicationStep(
            order=len(steps) + 1,
            description="Submit through Workspace on Grants.gov and confirm you receive a tracking number.",
            url="https://www.grants.gov/applicants/workspace-overview",
        )
    )
    return steps


#: The detail payload carries one of these blocks: ``synopsis`` for a posted
#: call, ``forecast`` for one an agency has only announced an intention to fund.
DETAIL_BLOCKS = ("synopsis", "forecast")

#: Field names differ between the two blocks. Left is synopsis, right is forecast.
_DATE_FIELDS = {
    "close": ("responseDate", "estimatedApplicationDueDate"),
    "posted": ("postingDate", "estimatedPostDate"),
}
_DESC_FIELDS = ("synopsisDesc", "forecastDesc")


def detail_block(detail: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return the payload's content block and whether it is a forecast."""
    for index, name in enumerate(DETAIL_BLOCKS):
        block = detail.get(name)
        if isinstance(block, dict):
            return block, bool(index)
    raise SourceShapeError(
        f"payload has none of {DETAIL_BLOCKS}; top-level keys were {sorted(detail)}"
    )


def normalise(detail: dict[str, Any]) -> Opportunity:
    """Turn one FetchOpportunity payload into an Opportunity record."""
    opportunity_id = detail.get("id") or detail.get("opportunityId")
    if opportunity_id is None:
        raise SourceShapeError("fetchOpportunity payload has no 'id'")

    try:
        synopsis, is_forecast = detail_block(detail)
    except SourceShapeError as exc:
        raise SourceShapeError(f"opportunity {opportunity_id}: {exc}") from exc

    def either(pair: tuple[str, str]) -> Any:
        """Read the field under whichever name this block uses."""
        return synopsis.get(pair[1 if is_forecast else 0]) or synopsis.get(pair[0])

    missing: list[str] = []

    def track(name: str, value: Any) -> Any:
        if value in (None, "", []):
            missing.append(name)
        return value

    number = str(detail.get("opportunityNumber") or opportunity_id)
    title = detail.get("opportunityTitle") or synopsis.get("opportunityTitle")
    if not title:
        raise SourceShapeError(f"opportunity {opportunity_id} has no title")

    funder = (
        synopsis.get("agencyName")
        or detail.get("agencyName")
        or detail.get("agencyCode")
        or "Unknown agency"
    )

    close_date = track("close_date", _parse_date(either(_DATE_FIELDS["close"])))
    ceiling = track("award_ceiling", _parse_money(synopsis.get("awardCeiling")))
    floor = track("award_floor", _parse_money(synopsis.get("awardFloor")))
    cost_share = synopsis.get("costSharing")
    if cost_share is None:
        missing.append("cost_share_required")

    forms = _forms(detail)
    if not forms:
        missing.append("forms")

    return Opportunity(
        id=f"{SOURCE}:{opportunity_id}",
        source=SOURCE,
        source_id=number,
        title=str(title).strip(),
        funder=str(funder).strip(),
        source_url=DETAIL_URL.format(id=opportunity_id),
        description=str(either(_DESC_FIELDS) or "").strip(),
        eligibility_text=str(synopsis.get("applicantEligibilityDesc") or "").strip(),
        applicant_codes=_applicant_codes(synopsis),
        award_floor=floor,
        award_ceiling=ceiling,
        total_pool=_parse_money(synopsis.get("estimatedFunding")),
        expected_awards=_parse_money(synopsis.get("expectedNumberOfAwards")),
        posted_date=_parse_date(either(_DATE_FIELDS["posted"])),
        close_date=close_date,
        is_forecast=is_forecast,
        jurisdiction="US",
        cost_share_required=bool(cost_share) if cost_share is not None else None,
        prerequisites=["SAM.gov registration with an active UEI", "Grants.gov account"],
        forms=forms,
        steps=_steps(str(opportunity_id), forms),
        fetched_at=utcnow(),
        missing_fields=sorted(set(missing)),
    )


def collect(
    keyword: str = "",
    limit: int = 50,
    statuses: str = "posted",
    client: httpx.Client | None = None,
) -> list[Opportunity]:
    """Search, fetch details, and normalise. One record per usable opportunity.

    ``client`` is injectable so the paging and error handling can be tested
    without touching the network.
    """
    owned = client is None
    client = client or httpx.Client(headers={"User-Agent": "Granter/0.1 (grant discovery)"})
    records: list[Opportunity] = []
    try:
        for hit in search(client, keyword=keyword, statuses=statuses, limit=limit):
            hit_id = hit.get("id")
            if hit_id is None:
                continue
            try:
                records.append(normalise(fetch_detail(client, hit_id)))
            except (SourceShapeError, httpx.HTTPError) as exc:
                # Skip loudly: a record we cannot normalise is dropped, never
                # patched up with defaults.
                print(f"  skipped opportunity {hit_id}: {exc}")
    finally:
        if owned:
            client.close()
    return records
