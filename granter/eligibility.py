"""Deterministic eligibility rules.

No model, no embedding, no inference. Every verdict traces to a comparison
between a stated applicant answer and a stated field on a retrieved call, and
every rule returns the note that explains itself in the UI.

Vocabulary used throughout:

* **blocker** -- the call, as retrieved, excludes this applicant. Hard filter.
* **caution** -- eligible, but something needs verifying or preparing.
* **unknown** -- the source document did not state the field. Never treated as
  a pass; it downgrades confidence and is shown to the user verbatim.
"""

from __future__ import annotations

from datetime import date

from .models import (
    Applicant,
    Confidence,
    Match,
    Note,
    Opportunity,
    RegistrationStatus,
    Verdict,
)
from .taxonomy import (
    AMBIGUOUS_CODES,
    APPLICANT_TYPE_TO_CODES,
    GRANTS_GOV_APPLICANT_CODES,
    NO_LEGAL_ENTITY_TYPES,
    ApplicantType,
)

#: A record older than this is still shown, but flagged for re-verification.
STALE_RECORD_DAYS = 7

#: SAM.gov registration routinely takes weeks. Below this many days to the
#: deadline, an applicant without a UEI is very unlikely to make it.
SAM_LEAD_TIME_DAYS = 30

#: An applicant asking for more than this multiple of the ceiling is not a
#: near miss, it is a different project.
AMOUNT_NEAR_MISS_FACTOR = 2.0


# --- Individual / no-entity handling ---------------------------------------


def entity_advisories(applicant: Applicant) -> list[Note]:
    """The honest conversation an applicant with no legal entity is owed.

    Federal grant programmes are overwhelmingly awarded to organisations. Rather
    than return a list of calls this applicant cannot receive, Granter says so
    up front and names the routes that do exist.
    """
    if applicant.applicant_type not in NO_LEGAL_ENTITY_TYPES:
        return []

    if applicant.has_fiscal_sponsor:
        return [
            Note(
                kind="caution",
                field="applicant_type",
                text=(
                    "You are applying through a fiscal sponsor, so the sponsor is the "
                    "legal applicant. Results below are filtered for what your sponsor's "
                    "organisation type can receive, not for individuals. Confirm the "
                    "sponsor's registration status and that they will sign the application."
                ),
            )
        ]

    noun = (
        "an individual"
        if applicant.applicant_type is ApplicantType.INDIVIDUAL
        else "an unincorporated group"
    )
    return [
        Note(
            kind="blocker",
            field="applicant_type",
            text=(
                f"You are {noun} with no legal entity. Most government grant programmes "
                "are awarded to organisations only, and US federal grants are almost never "
                "available to individuals for personal needs. Granter will show only calls "
                "that explicitly name individuals as eligible applicants — expect this list "
                "to be short."
            ),
        ),
        Note(
            kind="caution",
            field="applicant_type",
            text=(
                "The routes that usually do work: (1) a fiscal sponsor — an existing "
                "501(c)(3) that receives the grant and administers it for your project; "
                "(2) scholarships and fellowships, which are a separate funding category "
                "from grants; (3) applying jointly with a partner organisation that holds "
                "the award. Incorporating is also possible but takes months."
            ),
        ),
    ]


# --- Individual rules -------------------------------------------------------


def _check_applicant_type(applicant: Applicant, opp: Opportunity) -> list[Note]:
    allowed = APPLICANT_TYPE_TO_CODES.get(applicant.applicant_type, frozenset())

    if not opp.applicant_codes:
        return [
            Note(
                kind="unknown",
                field="applicant_codes",
                text=(
                    "This call does not publish a structured list of eligible applicant "
                    "types. Read the eligibility section on the funder's page before "
                    "investing time."
                ),
            )
        ]

    exact = allowed & set(opp.applicant_codes)
    if exact:
        names = ", ".join(GRANTS_GOV_APPLICANT_CODES.get(c, c) for c in sorted(exact))
        notes = [
            Note(
                kind="match",
                field="applicant_codes",
                text=f"The call names your applicant type as eligible: {names}.",
            )
        ]
        if applicant.applicant_type is ApplicantType.INFORMAL_GROUP:
            notes.append(
                Note(
                    kind="caution",
                    field="applicant_type",
                    text=(
                        "This call is open to individuals, so one member of your group would "
                        "have to be the named applicant and hold the award personally. The "
                        "group itself cannot be the recipient."
                    ),
                )
            )
        return notes

    ambiguous = AMBIGUOUS_CODES & set(opp.applicant_codes)
    if ambiguous:
        names = ", ".join(GRANTS_GOV_APPLICANT_CODES.get(c, c) for c in sorted(ambiguous))
        return [
            Note(
                kind="caution",
                field="applicant_codes",
                text=(
                    f"The call lists '{names}' rather than naming your applicant type. "
                    "That may or may not include you — the answer is in the call's "
                    "additional eligibility text, which you need to read."
                ),
            )
        ]

    listed = ", ".join(
        GRANTS_GOV_APPLICANT_CODES.get(c, c) for c in sorted(opp.applicant_codes)
    )
    return [
        Note(
            kind="blocker",
            field="applicant_codes",
            text=f"This call is limited to: {listed}. Your applicant type is not among them.",
        )
    ]


def _check_deadline(opp: Opportunity, today: date) -> list[Note]:
    if opp.rolling:
        return [Note(kind="match", field="close_date", text="Applications are accepted on a rolling basis.")]

    remaining = opp.days_remaining(today)
    if remaining is None:
        return [
            Note(
                kind="unknown",
                field="close_date",
                text="No closing date was published for this call. Verify on the funder's page.",
            )
        ]
    if remaining < 0:
        return [
            Note(
                kind="blocker",
                field="close_date",
                text=f"The deadline passed on {opp.close_date.isoformat()}.",
            )
        ]
    if remaining <= 14:
        return [
            Note(
                kind="caution",
                field="close_date",
                text=(
                    f"Closes in {remaining} day(s), on {opp.close_date.isoformat()}. "
                    "That is a short runway for a first-time application."
                ),
            )
        ]
    return [
        Note(
            kind="match",
            field="close_date",
            text=f"Closes {opp.close_date.isoformat()} — {remaining} days remaining.",
        )
    ]


def _check_geography(applicant: Applicant, opp: Opportunity) -> list[Note]:
    countries = {c.upper() for c in applicant.effective_countries}

    if opp.eligible_work_countries:
        allowed = {c.upper() for c in opp.eligible_work_countries}
        if countries & allowed:
            return [
                Note(
                    kind="match",
                    field="eligible_work_countries",
                    text="The call covers the country where your work takes place.",
                )
            ]
        return [
            Note(
                kind="blocker",
                field="eligible_work_countries",
                text=(
                    f"Funded work must take place in: {', '.join(sorted(allowed))}. "
                    f"Yours takes place in {', '.join(sorted(countries))}."
                ),
            )
        ]

    if opp.jurisdiction and opp.jurisdiction.upper() not in countries:
        return [
            Note(
                kind="caution",
                field="jurisdiction",
                text=(
                    f"This is a {opp.jurisdiction.upper()} funder and your work is in "
                    f"{', '.join(sorted(countries))}. Some programmes fund overseas work and "
                    "some do not; the call text decides it."
                ),
            )
        ]
    return []


def _check_amount(applicant: Applicant, opp: Opportunity) -> list[Note]:
    want = applicant.amount_sought
    if want is None:
        return []

    if opp.award_ceiling is not None and want > opp.award_ceiling:
        over = want / opp.award_ceiling
        kind = "blocker" if over > AMOUNT_NEAR_MISS_FACTOR else "caution"
        return [
            Note(
                kind=kind,
                field="award_ceiling",
                text=(
                    f"You are seeking ${want:,} but awards are capped at "
                    f"${opp.award_ceiling:,}."
                ),
            )
        ]

    if opp.award_floor is not None and want < opp.award_floor:
        # Asking far below the floor is not a project that needs rescoping, it
        # is a different programme -- treat it the same way as a far over-ask.
        far_below = want * AMOUNT_NEAR_MISS_FACTOR < opp.award_floor
        return [
            Note(
                kind="blocker" if far_below else "caution",
                field="award_floor",
                text=(
                    f"You are seeking ${want:,}; the smallest award is ${opp.award_floor:,}. "
                    + (
                        "This call funds work at a scale far larger than your project."
                        if far_below
                        else "You may need to widen the project scope."
                    )
                ),
            )
        ]

    if opp.award_ceiling is None and opp.award_floor is None:
        return [
            Note(
                kind="unknown",
                field="award_ceiling",
                text="No award range was published. Check the call before sizing a budget.",
            )
        ]
    return [
        Note(
            kind="match",
            field="award_ceiling",
            text=f"Your ${want:,} request fits this call's award range.",
        )
    ]


def _check_timeline(applicant: Applicant, opp: Opportunity, today: date) -> list[Note]:
    if applicant.project_start is None or opp.close_date is None or opp.rolling:
        return []
    if applicant.project_start < opp.close_date:
        return [
            Note(
                kind="caution",
                field="project_start",
                text=(
                    f"You want to start on {applicant.project_start.isoformat()}, before this "
                    f"call even closes ({opp.close_date.isoformat()}). Awards are typically "
                    "announced months after the deadline."
                ),
            )
        ]
    return []


def _check_prerequisites(applicant: Applicant, opp: Opportunity, today: date) -> list[Note]:
    notes: list[Note] = []
    needs_sam = opp.source == "grants_gov" or any(
        "sam" in p.lower() or "uei" in p.lower() for p in opp.prerequisites
    )

    if needs_sam and applicant.sam_uei_status is not RegistrationStatus.COMPLETE:
        remaining = opp.days_remaining(today)
        tight = remaining is not None and remaining < SAM_LEAD_TIME_DAYS
        notes.append(
            Note(
                kind="blocker" if tight else "caution",
                field="sam_uei_status",
                text=(
                    "You must have an active SAM.gov registration with a UEI before you can "
                    "submit. Registration commonly takes several weeks."
                    + (
                        f" With {remaining} days left, this is unlikely to complete in time."
                        if tight
                        else ""
                    )
                ),
            )
        )

    if opp.source == "grants_gov" and not applicant.grants_gov_account:
        notes.append(
            Note(
                kind="caution",
                field="grants_gov_account",
                text="You will also need a Grants.gov account linked to your organisation's UEI.",
            )
        )

    if opp.cost_share_required:
        notes.append(
            Note(
                kind="caution",
                field="cost_share_required",
                text="This call requires cost sharing or matching funds. Confirm you can cover it.",
            )
        )
    elif opp.cost_share_required is None:
        notes.append(
            Note(
                kind="unknown",
                field="cost_share_required",
                text="Cost-sharing requirements were not published. Check the call.",
            )
        )

    if applicant.grant_experience.value == "none":
        notes.append(
            Note(
                kind="caution",
                field="grant_experience",
                text=(
                    "You reported no prior grant experience. Many federal calls require "
                    "audited financials or evidence of managing an award — check the "
                    "capability requirements before committing time."
                ),
            )
        )

    for prereq in opp.prerequisites:
        # Skip the ones that already have a dedicated check above, so the same
        # requirement is not reported twice.
        low = prereq.lower()
        if "sam" in low or "uei" in low or "grants.gov" in low:
            continue
        notes.append(Note(kind="caution", field="prerequisites", text=f"Prerequisite: {prereq}"))

    return notes


def _check_freshness(opp: Opportunity) -> list[Note]:
    age = opp.record_age_days()
    if age > STALE_RECORD_DAYS:
        return [
            Note(
                kind="caution",
                field="fetched_at",
                text=(
                    f"This record was last checked {age} days ago "
                    f"({opp.fetched_at.date().isoformat()}). Deadlines change — verify against "
                    "the funder's page before relying on it."
                ),
            )
        ]
    return []


# --- Assembly ---------------------------------------------------------------


def _grade(notes: list[Note], opp: Opportunity) -> tuple[Verdict, Confidence]:
    blockers = [n for n in notes if n.kind == "blocker"]
    unknowns = [n for n in notes if n.kind == "unknown"]
    cautions = [n for n in notes if n.kind == "caution"]

    type_note = next((n for n in notes if n.field == "applicant_codes"), None)
    type_blocked = type_note is not None and type_note.kind == "blocker"
    expired = any(n.field == "close_date" and n.kind == "blocker" for n in notes)

    if type_blocked or expired:
        verdict = Verdict.INELIGIBLE
    elif blockers:
        # Eligible class, but something else rules it out for now -- exactly the
        # thing worth showing separately rather than hiding.
        verdict = Verdict.NEAR_MISS
    elif type_note is not None and type_note.kind == "match" and not unknowns:
        verdict = Verdict.ELIGIBLE
    else:
        verdict = Verdict.LIKELY

    if unknowns or opp.missing_fields:
        confidence = Confidence.LOW
    elif cautions or opp.record_age_days() > STALE_RECORD_DAYS:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.HIGH

    return verdict, confidence


def evaluate(applicant: Applicant, opp: Opportunity, today: date | None = None) -> Match:
    """Run every rule against one call and return the graded result."""
    today = today or date.today()

    notes: list[Note] = []
    notes += _check_applicant_type(applicant, opp)
    notes += _check_deadline(opp, today)
    notes += _check_geography(applicant, opp)
    notes += _check_amount(applicant, opp)
    notes += _check_timeline(applicant, opp, today)
    notes += _check_prerequisites(applicant, opp, today)
    notes += _check_freshness(opp)

    verdict, confidence = _grade(notes, opp)
    return Match(opportunity=opp, verdict=verdict, confidence=confidence, notes=notes)
