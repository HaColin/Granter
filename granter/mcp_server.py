"""Granter as an MCP server, so Claude can search grants without inventing them.

    python -m granter.mcp_server

The point of exposing this over MCP rather than letting a model answer funding
questions directly: every field in every result here came out of a document
retrieved from a funder, carries the URL it came from and the date it was
fetched, and was graded against published eligibility rules by code. A model
asked "what grants can I get?" will produce plausible programme names, amounts
and deadlines that do not exist. This tool cannot.

The tool descriptions say so explicitly, because the description is the only
thing the calling model reads before deciding what to do with the output.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import search, store
from .intake import to_applicant
from .models import Match
from .taxonomy import APPLICANT_TYPE_LABELS, SECTOR_LABELS, ApplicantType, Sector

server = MCPServer(
    name="granter",
    instructions=(
        "Searches real, currently-open funding opportunities retrieved from Grants.gov "
        "(US federal), the California Grants Portal, and the EU Funding & Tenders Portal.\n\n"
        "Every result traces to a retrieved source document and carries its official URL "
        "and the date it was checked. Never add grants, deadlines, award amounts or "
        "eligibility conclusions of your own to what these tools return, and never "
        "present a remembered programme name as a current opportunity — if it is not in "
        "the tool output, it was not found. Quote the source_url when reporting a result, "
        "and pass on the verify_before_applying warning."
    ),
)

MAX_RESULTS = 10


def _summarise(match: Match) -> dict[str, Any]:
    opportunity = match.opportunity
    return {
        "id": opportunity.id,
        "title": opportunity.title,
        "funder": opportunity.funder,
        "source": opportunity.source,
        "source_url": str(opportunity.source_url),
        "deadline": (
            "rolling" if opportunity.rolling
            else opportunity.close_date.isoformat() if opportunity.close_date
            else None
        ),
        "days_remaining": opportunity.days_remaining(),
        "is_forecast": opportunity.is_forecast,
        "award_floor": opportunity.award_floor,
        "award_ceiling": opportunity.award_ceiling,
        "verdict": match.verdict.value,
        "confidence": match.confidence.value,
        "why_it_matches": [n.text for n in match.notes if n.kind == "match"],
        "before_you_apply": [n.text for n in match.notes if n.kind == "caution"],
        "blockers": [n.text for n in match.notes if n.kind == "blocker"],
        "not_published_by_funder": opportunity.missing_fields,
        "record_checked_on": opportunity.fetched_at.date().isoformat(),
    }


@server.tool(
    description=(
        "Find funding opportunities an applicant is plausibly eligible for. Returns only "
        "opportunities retrieved from a funder's own published data, each with its "
        "official URL, deadline, and the specific reasons it matched or did not. "
        "Results are graded by deterministic rules, not by a model.\n\n"
        "Report what this returns and nothing more: do not supplement it with grants you "
        "recall, and do not soften or override a blocker. An empty result is a real "
        "answer meaning nothing currently open fits — say that rather than filling the "
        "gap. Always surface the 'advisories' field first; for an individual or informal "
        "group it explains why the list is short and what routes actually exist."
    ),
)
def search_grants(
    applicant_type: str,
    country: str,
    project_description: str,
    region: str = "",
    amount_sought: int | None = None,
    sectors: list[str] | None = None,
    team_size: int | None = None,
    grant_experience: str = "none",
    sam_uei_status: str = "none",
    has_fiscal_sponsor: bool = False,
    work_countries: list[str] | None = None,
    limit: int = MAX_RESULTS,
) -> str:
    """Search the retrieved corpus.

    applicant_type: one of individual, informal_group, nonprofit_501c3,
        nonprofit_other, for_profit_small, for_profit_other, academic_public,
        academic_private, government_state, government_local, tribal_government,
        tribal_organization, school_district, housing_authority.
    country: ISO 3166-1 alpha-2, e.g. US, DE.
    region: state or region code where relevant, e.g. CA.
    sectors: any of health, education, environment, arts, research,
        infrastructure, housing, food_agriculture, community_development,
        human_rights, technology, other.
    grant_experience: none, some, or extensive.
    sam_uei_status: none, in_progress, or complete (US federal prerequisite).
    """
    try:
        applicant = to_applicant({
            "applicant_type": applicant_type,
            "country": country,
            "region": region,
            "project_description": project_description,
            "amount_sought": amount_sought,
            "sectors": sectors or [],
            "team_size": team_size,
            "grant_experience": grant_experience,
            "sam_uei_status": sam_uei_status,
            "has_fiscal_sponsor": "yes" if has_fiscal_sponsor else "no",
            "work_countries": ", ".join(work_countries or []),
        })
    except (KeyError, ValueError) as exc:
        return json.dumps({
            "error": f"invalid intake value: {exc}",
            "valid_applicant_types": [t.value for t in ApplicantType],
            "valid_sectors": [s.value for s in Sector],
        })

    corpus = store.load()
    result = search.run(applicant, corpus)

    capped = max(1, min(limit, 25))
    return json.dumps({
        "advisories": [{"kind": n.kind, "text": n.text} for n in result.advisories],
        "searched": result.corpus_size,
        "corpus_last_updated": (
            result.corpus_fetched_at.date().isoformat() if result.corpus_fetched_at else None
        ),
        "sources_searched": sorted(corpus.sources()),
        "match_count": len(result.matches),
        "matches": [_summarise(m) for m in result.matches[:capped]],
        "near_miss_count": len(result.near_misses),
        "near_misses": [_summarise(m) for m in result.near_misses[:5]],
        "eligible_but_unrelated_count": len(result.unrelated),
        "not_searched": [
            {"name": r.name, "url": str(r.url), "access": r.access, "why": r.why}
            for r in result.referrals
        ],
        "verify_before_applying": (
            "Confirm every deadline and eligibility rule on the funder's own page before "
            "applying. Records are point-in-time snapshots of published data."
        ),
    }, indent=2)


@server.tool(
    description=(
        "Full detail for one opportunity returned by search_grants, including the ordered "
        "application steps, any published forms, and the funder's own eligibility text. "
        "Where a field says null or 'No Forms Found', the funder did not publish it — do "
        "not fill that gap with a plausible value."
    ),
)
def grant_details(grant_id: str) -> str:
    """Look up one opportunity by the id returned in search results."""
    corpus = store.load()
    record = next((r for r in corpus.records if r.id == grant_id), None)
    if record is None:
        return json.dumps({
            "error": f"no retrieved record with id {grant_id!r}",
            "hint": "ids come from search_grants; the corpus may have been refreshed since",
        })

    return json.dumps({
        "id": record.id,
        "title": record.title,
        "funder": record.funder,
        "source": record.source,
        "source_url": str(record.source_url),
        "official_number": record.source_id,
        "description": record.description,
        "eligibility_text": record.eligibility_text,
        "award_floor": record.award_floor,
        "award_ceiling": record.award_ceiling,
        "total_pool": record.total_pool,
        "expected_awards": record.expected_awards,
        "deadline": (
            "rolling" if record.rolling
            else record.close_date.isoformat() if record.close_date else None
        ),
        "is_forecast": record.is_forecast,
        "cost_share_required": record.cost_share_required,
        "prerequisites": record.prerequisites,
        "application_steps": [
            {
                "order": s.order,
                "description": s.description,
                "url": str(s.url) if s.url else None,
                "start_days_ahead": s.lead_time_days,
            }
            for s in sorted(record.steps, key=lambda s: s.order)
        ],
        "forms": [
            {"name": f.name, "url": str(f.url) if f.url else None}
            for f in record.forms
        ] or "No Forms Found",
        "not_published_by_funder": record.missing_fields,
        "record_checked_on": record.fetched_at.date().isoformat(),
    }, indent=2)


@server.tool(
    description=(
        "What data is loaded and how fresh it is. Check this before reporting that "
        "nothing was found: an empty corpus means nothing has been fetched yet, which is "
        "a different answer from 'no opportunity fits'."
    ),
)
def corpus_status() -> str:
    """Report corpus size, sources and age."""
    from collections import Counter

    corpus = store.load()
    return json.dumps({
        "total_records": len(corpus),
        "by_source": dict(Counter(r.source for r in corpus.records)),
        "last_updated": corpus.fetched_at.isoformat() if corpus.fetched_at else None,
        "is_empty": corpus.is_empty,
        "refresh_command": "python -m granter.ingest --source all --limit 2000 --replace",
        "sources_covered": {
            "grants_gov": "US federal grants",
            "ca_grants": "California state grants",
            "eu_portal": "EU Funding & Tenders Portal calls",
        },
        "not_covered": (
            "Private foundations, local and city funders, US states other than "
            "California, and funders in Africa, Asia and Latin America. Say so if a "
            "user's need falls outside what is covered."
        ),
    }, indent=2)


@server.tool(
    description=(
        "The applicant types and sectors search_grants accepts, with human labels. Use "
        "this to map a user's description onto valid values instead of guessing."
    ),
)
def intake_options() -> str:
    """List the valid enum values for search_grants."""
    return json.dumps({
        "applicant_types": {t.value: APPLICANT_TYPE_LABELS[t] for t in ApplicantType},
        "sectors": {s.value: SECTOR_LABELS[s] for s in Sector},
        "grant_experience": {
            "none": "No prior grants",
            "some": "Some, but no audited financials",
            "extensive": "Yes, including audited financials",
        },
        "sam_uei_status": {
            "none": "No SAM.gov registration",
            "in_progress": "Started, not finished",
            "complete": "Active SAM.gov registration with a UEI",
        },
        "note": (
            "An individual or informal group with no legal entity is eligible for very "
            "little government funding. Pass their real status rather than upgrading them "
            "to an organisation; the advisory explains the routes that do exist."
        ),
    }, indent=2)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
