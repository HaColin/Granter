"""End-to-end tests through the HTTP layer.

The corpus is patched with synthetic fixtures so the real ``data/`` file is never
read or written here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from granter import app as app_module
from granter.store import Corpus

from .fixtures import opportunity


@pytest.fixture
def client(monkeypatch):
    corpus = Corpus([
        opportunity(id="a", source_id="A-1"),
        opportunity(id="b", source_id="B-2", applicant_codes=["21"],
                    title="Individual artist fellowship", forms=[]),
    ])
    monkeypatch.setattr(app_module.store, "load", lambda: corpus)
    return TestClient(app_module.app)


def nonprofit_answers(**overrides) -> dict:
    answers = {
        "applicant_type": "nonprofit_501c3",
        "country": "US",
        "project_description": "We test drinking water quality in rural wells.",
        "sectors": ["environment"],
        "amount_sought": "80000",
        "grant_experience": "extensive",
        "sam_uei_status": "complete",
        "grants_gov_account": "yes",
    }
    answers.update(overrides)
    return answers


def test_survey_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "What is applying for the grant?" in response.text


def test_results_page_shows_a_match_with_its_source_link(client):
    response = client.post("/results", data=nonprofit_answers())
    assert response.status_code == 200
    assert "Community water quality monitoring" in response.text
    assert "https://example.org/call/1" in response.text
    assert "Eligible" in response.text


def test_results_page_shows_no_forms_found_when_the_source_has_none(client):
    response = client.post("/results", data=nonprofit_answers(applicant_type="individual"))
    assert "No Forms Found" in response.text


def test_individual_sees_the_advisory_before_any_list(client):
    response = client.post("/results", data=nonprofit_answers(applicant_type="individual"))
    text = response.text
    assert "fiscal sponsor" in text
    # The advisory precedes the results heading.
    assert text.index("fiscal sponsor") < text.index("Matches (")


def test_results_page_names_the_paywalled_sources_it_did_not_search(client):
    response = client.post("/results", data=nonprofit_answers())
    assert "Candid" in response.text
    assert "African Development Bank" in response.text
    assert "Not searched" in response.text


def test_every_result_carries_a_last_checked_date(client):
    response = client.post("/results", data=nonprofit_answers())
    assert "Record last checked" in response.text


def test_bad_applicant_type_returns_the_form_not_a_crash(client):
    response = client.post("/results", data=nonprofit_answers(applicant_type="wizard"))
    assert response.status_code == 400
    assert "Could not read your answers" in response.text


def test_health_reports_corpus_state(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok" and payload["corpus_size"] == 2
