"""Core data model.

Two rules govern every field here:

1. A field is ``None`` when the source document did not state it. There is no
   default that stands in for missing data, because a guessed award ceiling or
   deadline is indistinguishable from a fabricated one downstream.
2. Every :class:`Opportunity` carries ``source_url`` and ``fetched_at`` so that
   any claim in the UI can be traced to a retrieved document and dated.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl

from .taxonomy import ApplicantType, Sector


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --- Opportunity ------------------------------------------------------------


class FormReference(BaseModel):
    """A form or document the applicant must complete."""

    name: str
    url: HttpUrl | None = None
    required: bool | None = None


class ApplicationStep(BaseModel):
    order: int
    description: str
    url: HttpUrl | None = None
    #: Steps that must be started well ahead of the deadline (registrations).
    lead_time_days: int | None = None


class Opportunity(BaseModel):
    """A funding call, normalised from one source, with provenance."""

    id: str
    source: str  # e.g. "grants_gov"
    source_id: str  # the funder's own opportunity number
    title: str
    funder: str
    #: The official funder page. Never an aggregator or paid intermediary.
    source_url: HttpUrl

    description: str = ""
    eligibility_text: str = ""

    #: Grants.gov eligible-applicant codes, verbatim from the source.
    applicant_codes: list[str] = Field(default_factory=list)

    award_floor: int | None = None
    award_ceiling: int | None = None
    total_pool: int | None = None
    expected_awards: int | None = None

    posted_date: date | None = None
    close_date: date | None = None
    #: Some calls accept applications continuously.
    rolling: bool = False
    #: A forecast is an announced intention to fund, not an open call. Its dates
    #: and amounts are the agency's estimates and routinely change.
    is_forecast: bool = False

    #: ISO 3166-1 alpha-2 of the funder's jurisdiction, e.g. "US".
    jurisdiction: str | None = None
    #: State or region the funder is limited to, e.g. "CA". Set by sources that
    #: are definitionally sub-national; left None by national ones.
    region: str | None = None
    #: Countries the funded work may take place in, when the call states it.
    eligible_work_countries: list[str] = Field(default_factory=list)

    sectors: list[Sector] = Field(default_factory=list)

    cost_share_required: bool | None = None
    #: Registrations that gate submission, e.g. "SAM.gov UEI", "Grants.gov".
    prerequisites: list[str] = Field(default_factory=list)

    forms: list[FormReference] = Field(default_factory=list)
    steps: list[ApplicationStep] = Field(default_factory=list)

    fetched_at: datetime
    #: Fields the normaliser could not populate from the source document.
    missing_fields: list[str] = Field(default_factory=list)
    #: Values the source did publish but this code could not read. These are
    #: bugs here, not gaps in the source, and are kept distinct from
    #: ``missing_fields`` so they can be found and fixed.
    parse_warnings: list[str] = Field(default_factory=list)

    def days_remaining(self, today: date | None = None) -> int | None:
        if self.close_date is None:
            return None
        return (self.close_date - (today or date.today())).days

    def is_open(self, today: date | None = None) -> bool:
        if self.rolling:
            return True
        remaining = self.days_remaining(today)
        return remaining is None or remaining >= 0

    def record_age_days(self, now: datetime | None = None) -> int:
        return ((now or utcnow()) - self.fetched_at).days


# --- Applicant intake -------------------------------------------------------


class RegistrationStatus(str, Enum):
    NONE = "none"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class GrantExperience(str, Enum):
    NONE = "none"
    SOME = "some"  # has received grants, no audited financials
    EXTENSIVE = "extensive"  # audited financials, federal awards before


class Applicant(BaseModel):
    """The answers from the intake survey."""

    applicant_type: ApplicantType
    #: Free-text registration detail, e.g. an EIN, charity number, or "".
    legal_registration: str = ""
    has_fiscal_sponsor: bool = False

    country: str  # ISO 3166-1 alpha-2, where the applicant is based
    region: str = ""
    work_countries: list[str] = Field(default_factory=list)

    sectors: list[Sector] = Field(default_factory=list)
    project_description: str = ""

    amount_sought: int | None = None
    project_start: date | None = None
    project_duration_months: int | None = None

    team_size: int | None = None
    grant_experience: GrantExperience = GrantExperience.NONE

    #: SAM.gov / UEI for US federal, PIC for EU, etc.
    sam_uei_status: RegistrationStatus = RegistrationStatus.NONE
    grants_gov_account: bool = False

    @property
    def effective_countries(self) -> list[str]:
        """Where the work happens, falling back to where the applicant is."""
        return self.work_countries or [self.country]

    @property
    def is_legal_entity(self) -> bool:
        from .taxonomy import NO_LEGAL_ENTITY_TYPES

        return self.applicant_type not in NO_LEGAL_ENTITY_TYPES


# --- Match output -----------------------------------------------------------


class Verdict(str, Enum):
    ELIGIBLE = "eligible"  # the call names this applicant class
    LIKELY = "likely"  # matched, but via an ambiguous code or missing data
    NEAR_MISS = "near_miss"  # eligible class, but some other criterion is off
    INELIGIBLE = "ineligible"  # excluded by an explicit statement in the call


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Note(BaseModel):
    """One reason, tied to the field it came from."""

    kind: str  # "match" | "blocker" | "caution" | "unknown"
    text: str
    field: str | None = None


class Match(BaseModel):
    opportunity: Opportunity
    verdict: Verdict
    confidence: Confidence
    score: float = 0.0
    #: How much the call's text overlaps the project description, 0..1.
    relevance: float = 0.0
    #: Share of the project's distinct terms that appear in the call, 0..1.
    #: Low coverage with a high BM25 score means one rare word matched, which
    #: is not the same as the call being about your project.
    term_coverage: float = 0.0
    notes: list[Note] = Field(default_factory=list)

    def notes_of(self, kind: str) -> list[Note]:
        return [n for n in self.notes if n.kind == kind]


class Referral(BaseModel):
    """A source Granter cannot search, surfaced honestly instead."""

    name: str
    url: HttpUrl
    why: str
    access: str  # "subscription" | "library" | "institutional"


class SearchResult(BaseModel):
    matches: list[Match] = Field(default_factory=list)
    near_misses: list[Match] = Field(default_factory=list)
    #: Guidance shown before any list, e.g. the individual-eligibility talk.
    advisories: list[Note] = Field(default_factory=list)
    referrals: list[Referral] = Field(default_factory=list)
    corpus_size: int = 0
    corpus_fetched_at: datetime | None = None
