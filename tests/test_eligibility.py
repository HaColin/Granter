"""Rules-engine tests. Fixtures are synthetic — see tests/fixtures.py."""

from __future__ import annotations

from datetime import timedelta

import pytest

from granter.eligibility import entity_advisories, evaluate
from granter.models import Confidence, RegistrationStatus, Verdict
from granter.taxonomy import ApplicantType

from .fixtures import TODAY, applicant, opportunity


def kinds(match) -> set[str]:
    return {n.kind for n in match.notes}


# --- applicant type ---------------------------------------------------------


def test_named_applicant_type_is_eligible():
    match = evaluate(applicant(sam_uei_status=RegistrationStatus.COMPLETE,
                               grants_gov_account=True,
                               grant_experience="extensive"),
                     opportunity(), today=TODAY)
    assert match.verdict is Verdict.ELIGIBLE
    assert match.confidence is Confidence.HIGH


def test_unlisted_applicant_type_is_ineligible():
    match = evaluate(applicant(applicant_type=ApplicantType.FOR_PROFIT_SMALL),
                     opportunity(applicant_codes=["12"]), today=TODAY)
    assert match.verdict is Verdict.INELIGIBLE
    assert any(n.field == "applicant_codes" and n.kind == "blocker" for n in match.notes)


@pytest.mark.parametrize("code", ["25", "99"])
def test_ambiguous_codes_never_claim_eligibility(code):
    match = evaluate(applicant(), opportunity(applicant_codes=[code]), today=TODAY)
    assert match.verdict is not Verdict.ELIGIBLE
    assert any(n.field == "applicant_codes" and n.kind == "caution" for n in match.notes)


def test_missing_applicant_codes_downgrade_confidence():
    match = evaluate(applicant(), opportunity(applicant_codes=[]), today=TODAY)
    assert match.verdict is Verdict.LIKELY
    assert match.confidence is Confidence.LOW


# --- individuals ------------------------------------------------------------


def test_individual_excluded_from_organisation_only_call():
    match = evaluate(applicant(applicant_type=ApplicantType.INDIVIDUAL),
                     opportunity(applicant_codes=["12", "13"]), today=TODAY)
    assert match.verdict is Verdict.INELIGIBLE


def test_individual_matches_call_that_names_individuals():
    match = evaluate(applicant(applicant_type=ApplicantType.INDIVIDUAL),
                     opportunity(applicant_codes=["21"]), today=TODAY)
    assert match.verdict in (Verdict.ELIGIBLE, Verdict.LIKELY, Verdict.NEAR_MISS)
    assert any(n.kind == "match" and n.field == "applicant_codes" for n in match.notes)


def test_individual_gets_a_blocking_advisory_before_any_list():
    notes = entity_advisories(applicant(applicant_type=ApplicantType.INDIVIDUAL))
    assert notes and notes[0].kind == "blocker"
    combined = " ".join(n.text for n in notes).lower()
    assert "fiscal sponsor" in combined
    assert "scholarship" in combined


def test_informal_group_also_gets_the_advisory():
    notes = entity_advisories(applicant(applicant_type=ApplicantType.INFORMAL_GROUP))
    assert any(n.kind == "blocker" for n in notes)


def test_informal_group_sees_calls_open_to_individuals():
    """The only route open to them must not be filtered out of the list."""
    match = evaluate(applicant(applicant_type=ApplicantType.INFORMAL_GROUP),
                     opportunity(applicant_codes=["21"]), today=TODAY)
    assert match.verdict is not Verdict.INELIGIBLE
    assert any(
        n.field == "applicant_type" and "named applicant" in n.text for n in match.notes
    )


def test_informal_group_is_still_excluded_from_organisation_only_calls():
    match = evaluate(applicant(applicant_type=ApplicantType.INFORMAL_GROUP),
                     opportunity(applicant_codes=["12"]), today=TODAY)
    assert match.verdict is Verdict.INELIGIBLE


def test_fiscal_sponsor_changes_the_advisory_to_a_caution():
    notes = entity_advisories(
        applicant(applicant_type=ApplicantType.INDIVIDUAL, has_fiscal_sponsor=True)
    )
    assert notes and all(n.kind == "caution" for n in notes)


def test_organisations_get_no_entity_advisory():
    assert entity_advisories(applicant()) == []


# --- deadlines --------------------------------------------------------------


def test_expired_call_is_ineligible():
    match = evaluate(applicant(), opportunity(close_date=TODAY - timedelta(days=1)), today=TODAY)
    assert match.verdict is Verdict.INELIGIBLE


def test_missing_deadline_is_flagged_not_assumed():
    match = evaluate(applicant(), opportunity(close_date=None), today=TODAY)
    assert any(n.field == "close_date" and n.kind == "unknown" for n in match.notes)
    assert match.confidence is Confidence.LOW


def test_rolling_call_is_always_open():
    match = evaluate(applicant(), opportunity(close_date=None, rolling=True), today=TODAY)
    assert not any(n.field == "close_date" and n.kind == "blocker" for n in match.notes)


# --- money ------------------------------------------------------------------


def test_request_far_above_ceiling_is_a_near_miss():
    match = evaluate(applicant(amount_sought=1_000_000), opportunity(), today=TODAY)
    assert match.verdict is Verdict.NEAR_MISS


def test_request_slightly_above_ceiling_is_only_a_caution():
    match = evaluate(applicant(amount_sought=160_000), opportunity(), today=TODAY)
    assert match.verdict is not Verdict.NEAR_MISS
    assert any(n.field == "award_ceiling" and n.kind == "caution" for n in match.notes)


def test_request_far_below_floor_is_a_near_miss():
    match = evaluate(applicant(amount_sought=1_000), opportunity(), today=TODAY)
    assert any(n.field == "award_floor" and n.kind == "blocker" for n in match.notes)
    assert match.verdict is Verdict.NEAR_MISS


def test_request_slightly_below_floor_is_only_a_caution():
    match = evaluate(applicant(amount_sought=20_000), opportunity(), today=TODAY)
    assert any(n.field == "award_floor" and n.kind == "caution" for n in match.notes)
    assert match.verdict is not Verdict.NEAR_MISS


# --- geography --------------------------------------------------------------


def test_work_outside_stated_countries_is_blocked():
    match = evaluate(applicant(work_countries=["KE"]),
                     opportunity(eligible_work_countries=["US"]), today=TODAY)
    assert match.verdict is Verdict.NEAR_MISS
    assert any(n.field == "eligible_work_countries" and n.kind == "blocker" for n in match.notes)


def test_foreign_applicant_to_us_funder_is_a_caution_not_a_rejection():
    match = evaluate(applicant(country="GB"), opportunity(), today=TODAY)
    assert any(n.field == "jurisdiction" and n.kind == "caution" for n in match.notes)


# --- prerequisites ----------------------------------------------------------


def test_missing_sam_registration_near_a_deadline_blocks():
    match = evaluate(
        applicant(),
        opportunity(source="grants_gov", close_date=TODAY + timedelta(days=10)),
        today=TODAY,
    )
    assert any(n.field == "sam_uei_status" and n.kind == "blocker" for n in match.notes)
    assert match.verdict is Verdict.NEAR_MISS


def test_missing_sam_registration_with_time_is_only_a_caution():
    match = evaluate(
        applicant(),
        opportunity(source="grants_gov", close_date=TODAY + timedelta(days=200)),
        today=TODAY,
    )
    assert any(n.field == "sam_uei_status" and n.kind == "caution" for n in match.notes)


def test_a_prerequisite_is_not_reported_twice():
    match = evaluate(
        applicant(),
        opportunity(source="grants_gov",
                    prerequisites=["SAM.gov registration with an active UEI", "Grants.gov account"]),
        today=TODAY,
    )
    texts = [n.text for n in match.notes]
    assert sum("Grants.gov account" in t for t in texts) == 1
    assert sum("SAM.gov" in t for t in texts) == 1


def test_an_unrelated_prerequisite_is_still_surfaced():
    match = evaluate(applicant(), opportunity(prerequisites=["Letter of intent due 30 days prior"]),
                     today=TODAY)
    assert any("Letter of intent" in n.text for n in match.notes)


def test_unpublished_cost_share_is_reported_as_unknown():
    match = evaluate(applicant(), opportunity(cost_share_required=None), today=TODAY)
    assert any(n.field == "cost_share_required" and n.kind == "unknown" for n in match.notes)


def test_stale_record_is_flagged_for_reverification():
    from datetime import timedelta as td

    from granter.models import utcnow

    match = evaluate(applicant(), opportunity(fetched_at=utcnow() - td(days=30)), today=TODAY)
    assert any(n.field == "fetched_at" and n.kind == "caution" for n in match.notes)
    assert match.confidence is not Confidence.HIGH
