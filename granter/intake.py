"""The intake survey.

Eleven fields, declared once and rendered by the app. Branching is expressed as
``show_if`` predicates over the answers already given, so no one is asked about
SAM.gov registration before they have said they are a US organisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from .models import Applicant, GrantExperience, RegistrationStatus
from .taxonomy import (
    APPLICANT_TYPE_LABELS,
    NO_LEGAL_ENTITY_TYPES,
    SECTOR_LABELS,
    ApplicantType,
    Sector,
)

Answers = dict[str, Any]


@dataclass(frozen=True)
class Question:
    name: str
    prompt: str
    kind: str  # "select" | "multiselect" | "text" | "textarea" | "number" | "date" | "bool"
    why: str
    options: list[tuple[str, str]] = field(default_factory=list)
    required: bool = False
    placeholder: str = ""
    show_if: Callable[[Answers], bool] | None = None

    def visible(self, answers: Answers) -> bool:
        return self.show_if is None or self.show_if(answers)


def _is_us(answers: Answers) -> bool:
    return str(answers.get("country", "")).upper() == "US"


def _no_entity(answers: Answers) -> bool:
    value = answers.get("applicant_type")
    try:
        return ApplicantType(value) in NO_LEGAL_ENTITY_TYPES
    except ValueError:
        return False


QUESTIONS: list[Question] = [
    Question(
        name="applicant_type",
        prompt="What is applying for the grant?",
        kind="select",
        why="This is the hardest eligibility filter — most calls name specific applicant types.",
        options=[(t.value, APPLICANT_TYPE_LABELS[t]) for t in ApplicantType],
        required=True,
    ),
    Question(
        name="has_fiscal_sponsor",
        prompt="Do you have a fiscal sponsor who would receive the funds?",
        kind="bool",
        why=(
            "Without a legal entity you cannot hold most grants yourself. A fiscal sponsor "
            "changes who the legal applicant is, and with it what you are eligible for."
        ),
        show_if=_no_entity,
    ),
    Question(
        name="legal_registration",
        prompt="Registration number (EIN, charity number, company number)",
        kind="text",
        why="Funders verify legal status; having the number to hand also speeds up eligibility checks.",
        placeholder="e.g. 12-3456789",
        show_if=lambda a: not _no_entity(a),
    ),
    Question(
        name="country",
        prompt="Where is the applicant based? (ISO country code)",
        kind="text",
        why="Determines which funder pools apply at all.",
        placeholder="US",
        required=True,
    ),
    Question(
        name="region",
        prompt="State / region",
        kind="text",
        why="Many calls are limited to particular states or regions.",
        placeholder="e.g. CA",
    ),
    Question(
        name="work_countries",
        prompt="Where will the work happen? (comma-separated codes; leave blank if the same)",
        kind="text",
        why="Where the work happens often matters more than where you are registered.",
        placeholder="US, MX",
    ),
    Question(
        name="sectors",
        prompt="What field is the project in?",
        kind="multiselect",
        why="Used to rank calls by subject fit.",
        options=[(s.value, SECTOR_LABELS[s]) for s in Sector],
    ),
    Question(
        name="project_description",
        prompt="Describe the project in a few sentences.",
        kind="textarea",
        why="Matched against the text of each call. Concrete nouns work better than mission language.",
        placeholder="What you will do, for whom, and where.",
        required=True,
    ),
    Question(
        name="amount_sought",
        prompt="How much funding do you need? (USD)",
        kind="number",
        why="Filters out calls whose award range cannot fit your project.",
        placeholder="50000",
    ),
    Question(
        name="project_start",
        prompt="When would the project start?",
        kind="date",
        why="Awards are announced months after a deadline; this catches timelines that cannot work.",
    ),
    Question(
        name="project_duration_months",
        prompt="How many months will it run?",
        kind="number",
        why="Calls specify a project period your timeline has to fit inside.",
        placeholder="12",
    ),
    Question(
        name="team_size",
        prompt="How many people are on the team or in the organisation?",
        kind="number",
        why="Some calls set capacity minimums.",
        placeholder="4",
    ),
    Question(
        name="grant_experience",
        prompt="Have you managed grant funding before?",
        kind="select",
        why="Some calls are first-time-applicant friendly; others require audited financials.",
        options=[
            (GrantExperience.NONE.value, "No prior grants"),
            (GrantExperience.SOME.value, "Some — but no audited financials"),
            (GrantExperience.EXTENSIVE.value, "Yes, including audited financials"),
        ],
    ),
    Question(
        name="sam_uei_status",
        prompt="Do you have an active SAM.gov registration and UEI?",
        kind="select",
        why="A prerequisite for every US federal grant, and it takes weeks to obtain.",
        options=[
            (RegistrationStatus.NONE.value, "No"),
            (RegistrationStatus.IN_PROGRESS.value, "Started, not finished"),
            (RegistrationStatus.COMPLETE.value, "Yes, active"),
        ],
        show_if=_is_us,
    ),
    Question(
        name="grants_gov_account",
        prompt="Do you have a Grants.gov account?",
        kind="bool",
        why="Needed to submit; separate from SAM.gov.",
        show_if=_is_us,
    ),
]


def visible_questions(answers: Answers) -> list[Question]:
    return [q for q in QUESTIONS if q.visible(answers)]


# --- Parsing ----------------------------------------------------------------


def _int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _codes(value: Any) -> list[str]:
    if not value:
        return []
    return [part.strip().upper() for part in str(value).split(",") if part.strip()]


def to_applicant(answers: Answers) -> Applicant:
    """Build an Applicant from raw form answers, ignoring hidden branches."""
    visible = {q.name for q in visible_questions(answers)}

    def get(name: str, default: Any = "") -> Any:
        return answers.get(name, default) if name in visible else default

    sectors_raw = answers.get("sectors") or []
    if isinstance(sectors_raw, str):
        sectors_raw = [sectors_raw]

    return Applicant(
        applicant_type=ApplicantType(answers["applicant_type"]),
        legal_registration=str(get("legal_registration") or ""),
        has_fiscal_sponsor=_bool(get("has_fiscal_sponsor")),
        country=str(answers.get("country", "")).strip().upper() or "US",
        region=str(get("region") or "").strip(),
        work_countries=_codes(get("work_countries")),
        sectors=[Sector(s) for s in sectors_raw if s in Sector._value2member_map_],
        project_description=str(answers.get("project_description") or "").strip(),
        amount_sought=_int(get("amount_sought")),
        project_start=_date(get("project_start")),
        project_duration_months=_int(get("project_duration_months")),
        team_size=_int(get("team_size")),
        grant_experience=GrantExperience(get("grant_experience") or GrantExperience.NONE.value),
        sam_uei_status=RegistrationStatus(get("sam_uei_status") or RegistrationStatus.NONE.value),
        grants_gov_account=_bool(get("grants_gov_account")),
    )
