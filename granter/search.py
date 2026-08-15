"""Orchestration: intake answers in, ranked and separated results out."""

from __future__ import annotations

from datetime import date

from .eligibility import entity_advisories, evaluate
from .models import Applicant, Match, Note, Referral, SearchResult, Verdict
from .ranking import score_matches
from .store import Corpus
from .taxonomy import ApplicantType

#: Sources Granter does not search, named rather than silently ignored so the
#: user knows what was not covered. Two kinds: paywalled indexes, and public
#: funders that publish no machine-readable feed. The second kind is free to
#: browse -- the only barrier is that it has to be done by hand.
REFERRALS: list[Referral] = [
    Referral(
        name="African Development Bank",
        url="https://www.afdb.org/en/projects-and-operations",
        why=(
            "Africa's main development funder. Publishes projects and procurement "
            "notices on its own site with no open feed, so it has to be browsed directly."
        ),
        access="public, no API",
    ),
    Referral(
        name="Asian Development Bank",
        url="https://www.adb.org/projects",
        why="Development funding across Asia and the Pacific; project listings are web-only.",
        access="public, no API",
    ),
    Referral(
        name="Inter-American Development Bank",
        url="https://www.iadb.org/en/projects",
        why="Development funding across Latin America and the Caribbean.",
        access="public, no API",
    ),
    Referral(
        name="UN Partner Portal",
        url="https://www.unpartnerportal.org",
        why=(
            "Calls for expressions of interest from UN agencies, aimed at NGOs. "
            "Requires registration, so it cannot be indexed from outside."
        ),
        access="public, registration required",
    ),
    Referral(
        name="GlobalGiving",
        url="https://www.globalgiving.org",
        why="Crowdfunding and grant programmes for community organisations worldwide.",
        access="public, no API",
    ),
    Referral(
        name="Candid / Foundation Directory",
        url="https://candid.org",
        why="The main index of US private and family foundations, which no free API covers.",
        access="library",
    ),
    Referral(
        name="Devex Funding",
        url="https://www.devex.com/funding",
        why="International development and humanitarian funding, including donor pipelines.",
        access="subscription",
    ),
    Referral(
        name="GrantStation",
        url="https://grantstation.com",
        why="US state-level and private funders that publish no machine-readable feed.",
        access="subscription",
    ),
    Referral(
        name="Pivot-RP",
        url="https://pivot.proquest.com",
        why="Academic and fellowship funding; often free through a university library.",
        access="institutional",
    ),
    Referral(
        name="Research Professional",
        url="https://www.researchprofessional.com",
        why="Research funding across the UK and EU.",
        access="institutional",
    ),
]

FREE_ACCESS_NOTE = (
    "Every result Granter returns is free to apply for. If a site asks you to pay for "
    "access to a government grant application or for the information itself, it is a "
    "middleman — the official pages linked here cost nothing."
)


def effective_applicant(applicant: Applicant) -> Applicant:
    """The legal applicant, which is the sponsor when there is a fiscal sponsor.

    An individual with a fiscal sponsor is not applying as an individual: the
    sponsoring 501(c)(3) is the legal applicant and its eligibility governs.
    """
    if applicant.has_fiscal_sponsor and not applicant.is_legal_entity:
        return applicant.model_copy(update={"applicant_type": ApplicantType.NONPROFIT_501C3})
    return applicant


def run(applicant: Applicant, corpus: Corpus, today: date | None = None) -> SearchResult:
    legal = effective_applicant(applicant)

    graded = [evaluate(legal, opp, today=today) for opp in corpus.records]

    matches: list[Match] = [m for m in graded if m.verdict in (Verdict.ELIGIBLE, Verdict.LIKELY)]
    near: list[Match] = [m for m in graded if m.verdict is Verdict.NEAR_MISS]

    score_matches(legal, matches)
    score_matches(legal, near)

    advisories = entity_advisories(applicant)
    if corpus.is_empty:
        advisories.append(
            Note(
                kind="blocker",
                field="corpus",
                text=(
                    "No opportunity records are loaded. Granter will not show results it has "
                    "not retrieved. Run `python -m granter.ingest` to fetch live calls from "
                    "Grants.gov first."
                ),
            )
        )

    return SearchResult(
        matches=matches,
        near_misses=near,
        advisories=advisories,
        referrals=REFERRALS,
        corpus_size=len(corpus),
        corpus_fetched_at=corpus.fetched_at,
    )
