"""MCP surface tests.

The tool descriptions are as much a part of this interface as the schemas: they
are the only thing the calling model reads before deciding what to do with the
output, and the whole point of exposing Granter this way is to stop a model
answering funding questions from memory.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from granter import mcp_server
from granter.store import Corpus

from .fixtures import opportunity


@pytest.fixture
def corpus(monkeypatch):
    records = [
        opportunity(id="a", source_id="A-1"),
        opportunity(id="b", source_id="B-2", applicant_codes=["21"],
                    title="Individual artist fellowship", forms=[]),
    ]
    monkeypatch.setattr(mcp_server.store, "load", lambda: Corpus(records))
    return records


def call(tool, **kwargs) -> dict:
    return json.loads(tool(**kwargs))


# --- surface ----------------------------------------------------------------


def test_the_expected_tools_are_exposed():
    names = {t.name for t in asyncio.run(mcp_server.server.list_tools())}
    assert names == {"search_grants", "grant_details", "corpus_status", "intake_options"}


def test_every_tool_description_warns_against_inventing_grants():
    tools = asyncio.run(mcp_server.server.list_tools())
    described = " ".join(t.description or "" for t in tools).lower()
    assert "retrieved" in described
    assert "do not" in described or "never" in described


def test_the_server_instructions_forbid_adding_remembered_grants():
    instructions = (mcp_server.server.instructions or "").lower()
    assert "never add grants" in instructions
    assert "remembered" in instructions


# --- search -----------------------------------------------------------------


def test_search_returns_source_urls_and_check_dates(corpus):
    out = call(mcp_server.search_grants,
               applicant_type="nonprofit_501c3", country="US",
               project_description="Rural drinking water quality testing.")
    assert out["match_count"] >= 1
    for match in out["matches"]:
        assert match["source_url"].startswith("https://")
        assert match["record_checked_on"]
    assert "confirm every deadline" in out["verify_before_applying"].lower()


def test_search_reports_what_the_funder_did_not_publish(corpus):
    out = call(mcp_server.search_grants,
               applicant_type="nonprofit_501c3", country="US",
               project_description="Rural drinking water quality testing.")
    assert "not_published_by_funder" in out["matches"][0]


def test_an_individual_gets_the_advisory_first(corpus):
    out = call(mcp_server.search_grants,
               applicant_type="individual", country="US",
               project_description="Rural drinking water quality testing.")
    assert out["advisories"]
    assert out["advisories"][0]["kind"] == "blocker"
    assert "fiscal sponsor" in " ".join(a["text"] for a in out["advisories"])


def test_an_invalid_value_returns_the_valid_ones(corpus):
    out = call(mcp_server.search_grants, applicant_type="wizard", country="US",
               project_description="x")
    assert "error" in out
    assert "nonprofit_501c3" in out["valid_applicant_types"]


def test_search_names_the_sources_it_did_not_cover(corpus):
    out = call(mcp_server.search_grants,
               applicant_type="nonprofit_501c3", country="US",
               project_description="Rural drinking water quality testing.")
    names = {r["name"] for r in out["not_searched"]}
    assert "African Development Bank" in names
    assert "Candid / Foundation Directory" in names


def test_results_are_capped_so_a_model_is_not_flooded(corpus):
    out = call(mcp_server.search_grants,
               applicant_type="nonprofit_501c3", country="US",
               project_description="Rural drinking water quality testing.", limit=99)
    assert len(out["matches"]) <= 25


# --- details ----------------------------------------------------------------


def test_details_returns_steps_and_the_funder_s_own_text(corpus):
    out = call(mcp_server.grant_details, grant_id="a")
    assert out["application_steps"] or out["eligibility_text"]
    assert out["source_url"].startswith("https://")


def test_details_says_no_forms_found_rather_than_inventing_one(corpus):
    assert call(mcp_server.grant_details, grant_id="b")["forms"] == "No Forms Found"


def test_an_unknown_id_is_an_error_not_a_guess(corpus):
    out = call(mcp_server.grant_details, grant_id="does-not-exist")
    assert "error" in out and "no retrieved record" in out["error"]


# --- status -----------------------------------------------------------------


def test_status_distinguishes_empty_from_no_match(monkeypatch):
    monkeypatch.setattr(mcp_server.store, "load", lambda: Corpus([]))
    out = call(mcp_server.corpus_status)
    assert out["is_empty"] is True
    assert out["total_records"] == 0


def test_status_states_what_is_not_covered(corpus):
    out = call(mcp_server.corpus_status)
    assert "foundations" in out["not_covered"].lower()
    assert set(out["by_source"]) == {"test"}


def test_intake_options_lists_valid_values():
    out = call(mcp_server.intake_options)
    assert "nonprofit_501c3" in out["applicant_types"]
    assert "environment" in out["sectors"]
    assert "no legal entity" in out["note"]
