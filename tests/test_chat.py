"""Conversational intake tests.

Gemini is mocked throughout: these assert how its output is constrained and
merged, not what it says. The single most important property under test is that
the model can only fill in survey fields -- it can never put a grant, a
deadline or an eligibility verdict in front of a user.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from granter import app as app_module
from granter import chat
from granter.store import Corpus

from .fixtures import opportunity


def gemini_reply(reply: str, answers: dict, ready: bool = False) -> httpx.Response:
    payload = {"reply": reply, "ready": ready, "answers": answers}
    return httpx.Response(200, json={
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
    })


def client_returning(response_factory) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(response_factory))


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


# --- availability -----------------------------------------------------------


def test_chat_is_unavailable_without_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert chat.is_available() is False
    with pytest.raises(chat.ChatUnavailable, match="use the form"):
        chat.turn([], {})


def test_either_key_name_works(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert chat.is_available() is True


# --- extraction -------------------------------------------------------------


def test_a_turn_extracts_answers_in_the_form_s_own_vocabulary(api_key):
    def handler(request):
        return gemini_reply("Got it.", {
            "applicant_type": "for_profit_small", "country": "US", "region": "CA",
            "amount_sought": 50000, "sectors": ["environment", "technology"],
            "team_size": 2, "grants_gov_account": False,
            "project_description": "Solar water purifiers.",
        })

    reply, answers, ready = chat.turn(
        [{"role": "user", "text": "two-person CA startup, solar water purifiers, $50k"}],
        {}, client=client_returning(handler),
    )
    assert reply == "Got it."
    # Form-shaped values, ready to POST to /results unchanged.
    assert answers["amount_sought"] == "50000"
    assert answers["grants_gov_account"] == "no"
    assert answers["sectors"] == ["environment", "technology"]
    assert ready is False


def test_the_extracted_answers_parse_into_an_applicant(api_key):
    from granter.intake import to_applicant

    def handler(request):
        return gemini_reply("Searching.", {
            "applicant_type": "nonprofit_501c3", "country": "US", "region": "CA",
            "project_description": "Rural well testing.", "amount_sought": 80000,
        }, ready=True)

    _, answers, ready = chat.turn([], {}, client=client_returning(handler))
    assert ready is True
    applicant = to_applicant(answers)
    assert applicant.amount_sought == 80_000
    assert applicant.country == "US"


def test_a_later_turn_does_not_erase_an_earlier_answer(api_key):
    """Omitting a field is the model forgetting, not the user retracting."""
    existing = {"applicant_type": "nonprofit_501c3", "country": "US",
                "project_description": "Rural well testing."}

    def handler(request):
        return gemini_reply("Noted.", {"team_size": 4})

    _, answers, _ = chat.turn([], existing, client=client_returning(handler))
    assert answers["applicant_type"] == "nonprofit_501c3"
    assert answers["project_description"] == "Rural well testing."
    assert answers["team_size"] == "4"


def test_invalid_sector_values_are_dropped_rather_than_passed_through(api_key):
    def handler(request):
        return gemini_reply("ok", {"sectors": ["environment", "space_lasers"]})

    _, answers, _ = chat.turn([], {}, client=client_returning(handler))
    assert answers["sectors"] == ["environment"]


# --- the guarantee ----------------------------------------------------------


def test_the_model_cannot_submit_without_the_required_fields(api_key):
    """A confidently-wrong 'ready' must not run a search on an empty survey."""
    def handler(request):
        return gemini_reply("All done!", {"team_size": 3}, ready=True)

    _, _, ready = chat.turn([], {}, client=client_returning(handler))
    assert ready is False


def test_the_schema_sent_to_the_model_contains_only_survey_fields(api_key):
    """There is no field the model could use to return a grant or a verdict."""
    from granter.intake import QUESTIONS

    schema = chat._response_schema()
    answer_fields = set(schema["properties"]["answers"]["properties"])

    # Three top-level keys, none of which can carry grant content: a sentence to
    # say, a boolean, and the survey answers.
    assert set(schema["properties"]) == {"reply", "ready", "answers"}
    # And the answers are exactly the survey's own fields -- there is nowhere in
    # this schema for an opportunity, a deadline or a verdict to be returned.
    assert answer_fields == {q.name for q in QUESTIONS}


def test_the_prompt_forbids_naming_grants_and_judging_eligibility():
    assert "NEVER name a grant" in chat.SYSTEM_PROMPT
    assert "NEVER tell the user whether they are eligible" in chat.SYSTEM_PROMPT
    assert "administrator" in chat.SYSTEM_PROMPT  # injection attempts


def test_the_prompt_describes_the_real_survey_questions():
    """Generated from intake.QUESTIONS, so it cannot drift from the form."""
    guide = chat._field_guide()
    for name in ("applicant_type", "sam_uei_status", "project_description"):
        assert name in guide


# --- provider failures ------------------------------------------------------


def test_a_provider_error_is_reported_not_swallowed(api_key):
    def handler(request):
        return httpx.Response(429, text="quota exceeded")

    with pytest.raises(chat.ChatUnavailable, match="429"):
        chat.turn([], {}, client=client_returning(handler))


def test_a_non_json_response_is_rejected(api_key):
    def handler(request):
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "sorry, I'm just chatting"}]}}]
        })

    with pytest.raises(chat.ChatUnavailable, match="valid JSON"):
        chat.turn([], {}, client=client_returning(handler))


def test_an_unexpected_response_shape_is_rejected(api_key):
    def handler(request):
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    with pytest.raises(chat.ChatUnavailable, match="unexpected response shape"):
        chat.turn([], {}, client=client_returning(handler))


# --- routes -----------------------------------------------------------------


@pytest.fixture
def web(monkeypatch):
    monkeypatch.setattr(app_module.store, "load", lambda: Corpus([opportunity()]))
    return TestClient(app_module.app)


def test_chat_page_offers_the_form_when_there_is_no_key(web, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    response = web.get("/chat")
    assert response.status_code == 200
    assert "Chat is switched off" in response.text
    assert 'href="/"' in response.text


def test_chat_endpoint_returns_503_rather_than_a_fake_answer(web, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    response = web.post("/chat/message", json={"messages": [], "answers": {}})
    assert response.status_code == 503
    assert "form" in response.json()["error"]


def test_the_form_still_works_and_is_linked_from_chat(web, api_key):
    assert web.get("/").status_code == 200
    assert "Answer in a chat instead" in web.get("/").text
    assert "Prefer the form" in web.get("/chat").text


def test_chat_page_states_the_model_does_not_decide_eligibility(web, api_key):
    text = " ".join(web.get("/chat").text.split())  # collapse HTML line wrapping
    assert "only fills in the survey" in text
    assert "no access to grant data and never decides what you are eligible for" in text


def test_fields_the_survey_does_not_define_are_dropped(api_key):
    """The schema constrains the model; it does not bind it.

    A model that returns grant_name/deadline anyway must not have them merged
    into the answers, or the review panel would display a fabricated grant --
    through the one path designed to make that impossible.
    """
    def handler(request):
        return gemini_reply("You qualify for the Smith Foundation Grant!", {
            "country": "US",
            "grant_name": "Smith Foundation Grant",
            "deadline": "2026-12-01",
            "award_amount": "$50,000",
        })

    _, answers, _ = chat.turn([], {}, client=client_returning(handler))
    assert answers == {"country": "US"}
    assert "grant_name" not in answers


def test_the_review_panel_shows_nothing_without_a_survey_label():
    """Second gate: even if something reaches the answers dict, it is not shown."""
    smuggled = {"country": "US", "grant_name": "Smith Foundation Grant"}
    shown = dict(chat.humanise(smuggled))
    assert "Smith Foundation Grant" not in shown.values()
    assert not any("grant_name" in label for label in shown)
