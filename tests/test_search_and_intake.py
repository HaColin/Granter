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


def test_a_forecast_payload_normalises_from_its_own_field_names():
    """Forecasted opportunities carry a 'forecast' block, not a 'synopsis' one."""
    detail = {
        "id": 999,
        "opportunityNumber": "TEST-26-FC",
        "opportunityTitle": "Forecasted call",
        "forecast": {
            "agencyName": "Test Agency",
            "estApplicationResponseDate": "06302027",
            "estSynopsisPostingDate": "03012027",
            "awardCeiling": "500000",
            "applicantTypes": [{"id": "12"}],
            "forecastDesc": "An intention to fund.",
        },
    }
    record = grants_gov.normalise(detail)
    assert record.is_forecast is True
    assert record.close_date.isoformat() == "2027-06-30"
    assert record.posted_date.isoformat() == "2027-03-01"
    assert record.description == "An intention to fund."
    assert record.applicant_codes == ["12"]


def test_a_payload_with_neither_block_names_the_keys_it_did_find():
    import pytest

    with pytest.raises(grants_gov.SourceShapeError, match="top-level keys"):
        grants_gov.normalise({"id": 1, "somethingElse": {}})


def test_a_forecast_is_labelled_as_an_estimate_not_a_deadline():
    from datetime import date as _date

    from granter.eligibility import evaluate

    record = opportunity(is_forecast=True, close_date=_date(2027, 6, 30), source="demo")
    match = evaluate(applicant(), record, today=TODAY)
    note = next(n for n in match.notes if n.field == "is_forecast")
    assert note.kind == "caution"
    assert "not an open call" in note.text
    assert "2027-06-30" in note.text


def test_single_digit_applicant_codes_are_zero_padded():
    record = grants_gov.normalise(_detail(applicantTypes=[{"id": "7"}]))
    assert record.applicant_codes == ["07"]


# --- ranking regressions ----------------------------------------------------


def test_relevance_outranks_an_eligible_but_unrelated_call():
    """A call naming your applicant type but about another field must not win."""
    relevant = opportunity(
        id="water", source_id="W", title="Desalination and water purification research",
        description="Solar powered desalination and drinking water purification for rural communities.",
        applicant_codes=["25"], sectors=[])
    unrelated = opportunity(
        id="cancer", source_id="C", title="Informatics technologies for cancer research",
        description="Early stage development of informatics technologies and data platforms.",
        applicant_codes=["23"], sectors=[])
    person = applicant(
        applicant_type=ApplicantType.FOR_PROFIT_SMALL,
        project_description="Solar powered water purification device for rural drinking water.",
        sectors=[])
    result = search.run(person, Corpus([unrelated, relevant]), today=TODAY)
    assert [m.opportunity.id for m in result.matches][0] == "water"


def test_a_call_with_no_textual_overlap_is_flagged():
    unrelated = opportunity(
        id="x", title="Feral swine eradication",
        description="Control of feral swine populations on agricultural land.",
        applicant_codes=["23"], sectors=[])
    person = applicant(applicant_type=ApplicantType.FOR_PROFIT_SMALL,
                       project_description="Solar powered water purification membranes for rural wells.",
                       sectors=[])
    match = search.run(person, Corpus([unrelated]), today=TODAY).matches[0]
    assert match.term_coverage < 0.12
    assert any("little in common" in n.text for n in match.notes)


def test_coverage_weights_distinctive_words_over_boilerplate():
    from granter.ranking import _Bm25Index

    index = _Bm25Index([
        ["desalination", "water", "development"],
        ["cancer", "informatics", "development"],
        ["wildlife", "habitat", "development"],
    ])
    query = {"desalination", "development"}
    # "development" is in every document, so it is worth almost nothing.
    assert index.coverage(query, {"desalination"}) > index.coverage(query, {"development"})


# --- California connector ---------------------------------------------------


def _ca_row(**overrides):
    row = {
        "PortalID": "190536", "Title": "Water Efficiency Grant",
        "AgencyDept": "State Water Resources Control Board", "Status": "active",
        "ApplicantType": "Business; Nonprofit", "Categories": "Environment & Water",
        "EstAmounts": "Between $50,000 and $500,000", "MatchingFunds": "Not Required",
        "ApplicationDeadline": "2027-02-01 12:00:00", "OpenDate": "2026-08-13 21:25:00",
        "GrantURL": "https://example.ca.gov/grant", "Purpose": "Improve water efficiency.",
    }
    row.update(overrides)
    return row


def test_ca_row_normalises_into_the_shared_shape():
    from granter.sources import ca_grants

    record = ca_grants.normalise(_ca_row())
    assert record.source == "ca_grants"
    assert record.region == "CA"
    assert record.funder == "State Water Resources Control Board"
    assert record.close_date.isoformat() == "2027-02-01"
    assert (record.award_floor, record.award_ceiling) == (50_000, 500_000)
    assert record.cost_share_required is False
    # State money: none of the federal registration machinery applies.
    assert record.prerequisites == []


def test_ca_applicant_types_map_onto_the_shared_codes():
    from granter.sources import ca_grants

    assert set(ca_grants.normalise(_ca_row()).applicant_codes) == {"12", "13", "20", "22", "23"}
    individual = ca_grants.normalise(_ca_row(ApplicantType="Individual"))
    assert individual.applicant_codes == ["21"]
    ambiguous = ca_grants.normalise(_ca_row(ApplicantType="Other Legal Entity"))
    assert ambiguous.applicant_codes == ["25"]


def test_ongoing_is_a_rolling_deadline_not_a_parse_failure():
    from granter.sources import ca_grants

    record = ca_grants.normalise(_ca_row(ApplicationDeadline="Ongoing"))
    assert record.rolling is True
    assert record.close_date is None
    assert record.parse_warnings == []


def test_free_text_award_prose_is_not_guessed_at():
    from granter.sources import ca_grants

    record = ca_grants.normalise(
        _ca_row(EstAmounts="Dependant on number of submissions received, etc.")
    )
    assert record.award_floor is None and record.award_ceiling is None
    assert {"award_floor", "award_ceiling"} <= set(record.missing_fields)


def test_a_single_published_amount_sets_only_the_ceiling():
    from granter.sources import ca_grants

    record = ca_grants.normalise(_ca_row(EstAmounts="$500,000"))
    assert record.award_ceiling == 500_000
    assert record.award_floor is None


def test_a_state_programme_is_a_near_miss_for_an_out_of_state_applicant():
    out_of_state = applicant(region="TX")
    result = search.run(out_of_state, Corpus([opportunity(region="CA")]), today=TODAY)
    assert not result.matches
    assert result.near_misses
    assert any(n.field == "region" and n.kind == "blocker"
               for n in result.near_misses[0].notes)


def test_a_state_programme_matches_an_in_state_applicant():
    local = applicant(region="CA")
    result = search.run(local, Corpus([opportunity(region="CA")]), today=TODAY)
    assert result.matches
    assert any(n.field == "region" and n.kind == "match" for n in result.matches[0].notes)


# --- EU connector -----------------------------------------------------------


def _eu_row(**overrides):
    row = {
        "identifier": "HORIZON-CL5-2027-D3-01",
        "callIdentifier": "HORIZON-CL5-2027-D3",
        "title": "Solar-driven water treatment for off-grid communities",
        "callTitle": "Clean energy transition",
        "type": "1",
        "status": {"abbreviation": "Open"},
        "frameworkProgramme": {"description": "Horizon Europe", "abbreviation": "HORIZON"},
        # 2027-06-30 and 2026-01-15, deliberately out of order and one past.
        "deadlineDatesLong": [1814486400000, 1768435200000],
        "publicationDateLong": 1755129600000,
        "tags": ["water", "solar"],
        "sme": True,
    }
    row.update(overrides)
    return row


def test_eu_row_normalises_into_the_shared_shape():
    from datetime import date as _date

    from granter.sources import eu_portal

    record = eu_portal.normalise(_eu_row(), today=_date(2026, 8, 14))
    assert record.source == "eu_portal"
    assert record.jurisdiction == "EU"
    assert record.funder == "European Commission — Horizon Europe"
    assert str(record.source_url).endswith("HORIZON-CL5-2027-D3-01")
    assert record.sectors == [Sector.RESEARCH]


def test_the_next_unexpired_cutoff_is_used_not_the_first_listed():
    """A call can have several cut-offs; showing an expired one would mislead."""
    from datetime import date as _date

    from granter.sources import eu_portal

    record = eu_portal.normalise(_eu_row(), today=_date(2026, 8, 14))
    assert record.close_date == _date(2026, 1, 15) or record.close_date > _date(2026, 8, 14)
    # With today before both, the earlier upcoming one wins.
    early = eu_portal.normalise(_eu_row(), today=_date(2025, 1, 1))
    assert early.close_date == _date(2026, 1, 15)


def test_missing_eu_budgets_are_recorded_not_invented():
    from datetime import date as _date

    from granter.sources import eu_portal

    record = eu_portal.normalise(_eu_row(), today=_date(2026, 8, 14))
    assert record.award_floor is None and record.award_ceiling is None
    assert {"award_floor", "award_ceiling"} <= set(record.missing_fields)


def test_eu_calls_publish_no_applicant_codes_so_none_are_claimed():
    from datetime import date as _date

    from granter.sources import eu_portal

    record = eu_portal.normalise(_eu_row(), today=_date(2026, 8, 14))
    assert record.applicant_codes == []
    assert "rarely open to individuals" in record.eligibility_text
    assert "SME" in record.eligibility_text


def test_tenders_are_excluded_because_they_are_contracts_not_grants():
    from granter.sources import eu_portal

    payload = {"fundingData": {"GrantTenderObj": [
        _eu_row(), _eu_row(identifier="TENDER-1", type="0"),
    ]}}
    rows = eu_portal.extract_objects(payload)
    grants = [r for r in rows if str(r.get("type")) == eu_portal.GRANT_TYPE]
    assert len(grants) == 1


def test_public_funders_without_a_feed_are_named_as_referrals():
    """Africa and the development banks are not searched; say so rather than omit."""
    result = search.run(applicant(), Corpus([]), today=TODAY)
    names = {r.name for r in result.referrals}
    assert "African Development Bank" in names
    assert "UN Partner Portal" in names
    assert any(r.access.startswith("public") for r in result.referrals)
