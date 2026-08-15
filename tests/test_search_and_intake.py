"""Orchestration, ranking, intake parsing, and normaliser tests."""

from __future__ import annotations

from datetime import timedelta

from granter import search
from granter.intake import to_applicant, visible_questions
from granter.models import Verdict
from granter.ranking import score_matches
from granter.sources import grants_gov
from granter.store import Corpus, merge
from granter.taxonomy import ApplicantType, Sector

from .fixtures import TODAY, applicant, opportunity


# --- search -----------------------------------------------------------------


def test_near_misses_are_separated_from_matches():
    corpus = Corpus([
        opportunity(id="a", source_id="A"),
        opportunity(id="b", source_id="B", award_ceiling=1_000),
        opportunity(id="c", source_id="C", applicant_codes=["23"]),
    ])
    result = search.run(applicant(), corpus, today=TODAY)

    assert [m.opportunity.id for m in result.matches] == ["a"]
    assert [m.opportunity.id for m in result.near_misses] == ["b"]
    # The ineligible one appears in neither list.
    assert "c" not in {m.opportunity.id for m in result.matches + result.near_misses}


def test_empty_corpus_says_so_instead_of_returning_nothing_silently():
    result = search.run(applicant(), Corpus([]), today=TODAY)
    assert result.matches == []
    assert any("ingest" in n.text for n in result.advisories)


def test_paywalled_sources_are_named_not_silently_skipped():
    result = search.run(applicant(), Corpus([]), today=TODAY)
    names = {r.name for r in result.referrals}
    assert "Candid / Foundation Directory" in names
    assert all(str(r.url).startswith("https://") for r in result.referrals)


def test_fiscal_sponsor_is_evaluated_as_the_sponsoring_nonprofit():
    person = applicant(applicant_type=ApplicantType.INDIVIDUAL, has_fiscal_sponsor=True)
    legal = search.effective_applicant(person)
    assert legal.applicant_type is ApplicantType.NONPROFIT_501C3

    result = search.run(person, Corpus([opportunity(applicant_codes=["12"])]), today=TODAY)
    assert result.matches and result.matches[0].verdict is not Verdict.INELIGIBLE


def test_individual_without_sponsor_is_not_promoted():
    person = applicant(applicant_type=ApplicantType.INDIVIDUAL)
    assert search.effective_applicant(person).applicant_type is ApplicantType.INDIVIDUAL


# --- ranking ----------------------------------------------------------------


def test_sector_and_text_overlap_outrank_an_unrelated_call():
    relevant = opportunity(id="water", source_id="W")
    unrelated = opportunity(
        id="bridge",
        source_id="B",
        title="Highway bridge replacement",
        description="Funds replacement of structurally deficient highway bridges.",
        sectors=[Sector.INFRASTRUCTURE],
    )
    result = search.run(applicant(), Corpus([unrelated, relevant]), today=TODAY)
    assert [m.opportunity.id for m in result.matches][0] == "water"


def test_scoring_is_stable_with_a_single_match():
    matches = search.run(applicant(), Corpus([opportunity()]), today=TODAY).matches
    assert score_matches(applicant(), matches)[0].score >= 0


# --- intake -----------------------------------------------------------------


def test_branching_hides_sam_questions_outside_the_us():
    names = {q.name for q in visible_questions({"applicant_type": "nonprofit_501c3", "country": "GB"})}
    assert "sam_uei_status" not in names
    assert "grants_gov_account" not in names


def test_branching_shows_fiscal_sponsor_only_without_a_legal_entity():
    individual = {q.name for q in visible_questions({"applicant_type": "individual", "country": "US"})}
    nonprofit = {q.name for q in visible_questions({"applicant_type": "nonprofit_501c3", "country": "US"})}
    assert "has_fiscal_sponsor" in individual and "legal_registration" not in individual
    assert "has_fiscal_sponsor" not in nonprofit and "legal_registration" in nonprofit


def test_answers_to_hidden_questions_are_discarded():
    parsed = to_applicant({
        "applicant_type": "nonprofit_501c3",
        "country": "GB",
        "project_description": "x",
        "sam_uei_status": "complete",  # hidden for a GB applicant
        "has_fiscal_sponsor": "yes",  # hidden for an organisation
    })
    assert parsed.sam_uei_status.value == "none"
    assert parsed.has_fiscal_sponsor is False


def test_free_text_numbers_are_parsed_leniently():
    parsed = to_applicant({
        "applicant_type": "individual",
        "country": "us",
        "project_description": "x",
        "amount_sought": "$50,000",
        "work_countries": "us, mx",
    })
    assert parsed.amount_sought == 50_000
    assert parsed.work_countries == ["US", "MX"]
    assert parsed.country == "US"


# --- store ------------------------------------------------------------------


def test_merge_prefers_the_newer_record():
    old = opportunity(id="x", title="Old title")
    new = opportunity(id="x", title="New title")
    merged = merge([old], [new])
    assert len(merged) == 1 and merged[0].title == "New title"


# --- normaliser -------------------------------------------------------------


def _detail(**synopsis_overrides):
    synopsis = {
        "agencyName": "Test Agency",
        "responseDate": "12312026",
        "postingDate": "01152026",
        "awardCeiling": "150000",
        "awardFloor": "25000",
        "applicantTypes": [{"id": "12", "description": "Nonprofits"}],
        "synopsisDesc": "A description.",
        "applicantEligibilityDesc": "Nonprofits only.",
        "costSharing": False,
    }
    synopsis.update(synopsis_overrides)
    return {"id": 12345, "opportunityNumber": "TEST-26-001",
            "opportunityTitle": "Test call", "synopsis": synopsis}


def test_normalise_maps_the_published_fields():
    record = grants_gov.normalise(_detail())
    assert record.source_url.host == "www.grants.gov"
    assert record.applicant_codes == ["12"]
    assert record.award_ceiling == 150_000
    assert record.close_date.isoformat() == "2026-12-31"
    assert record.prerequisites and record.steps


def test_normalise_records_what_the_source_did_not_publish():
    record = grants_gov.normalise(_detail(awardCeiling=None, responseDate=None, costSharing=None))
    assert record.award_ceiling is None and record.close_date is None
    assert {"award_ceiling", "close_date", "cost_share_required"} <= set(record.missing_fields)


def test_no_package_means_no_forms_rather_than_a_guessed_link():
    record = grants_gov.normalise(_detail())
    assert record.forms == []
    assert "forms" in record.missing_fields


def test_unrecognised_payload_raises_instead_of_emitting_a_partial_record():
    import pytest

    with pytest.raises(grants_gov.SourceShapeError):
        grants_gov.normalise({"id": 1})  # no synopsis


def test_single_digit_applicant_codes_are_zero_padded():
    record = grants_gov.normalise(_detail(applicantTypes=[{"id": "7"}]))
    assert record.applicant_codes == ["07"]
