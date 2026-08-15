"""SYNTHETIC TEST FIXTURES — NOT GRANT DATA.

Every opportunity below is invented for the purpose of exercising the rules
engine. None of these calls exist. They live in ``tests/`` and are never loaded
by :mod:`granter.store`, so they cannot leak into a user-facing result list.

Real records only ever come from an ingest run against a live source.
"""

from __future__ import annotations

from datetime import date, timedelta

from granter.models import Applicant, FormReference, Opportunity, utcnow
from granter.taxonomy import ApplicantType, Sector

TODAY = date(2026, 8, 14)


def opportunity(**overrides) -> Opportunity:
    defaults = dict(
        id="test:1",
        source="test",
        source_id="TEST-001",
        title="Community water quality monitoring",
        funder="Test Agency",
        source_url="https://example.org/call/1",
        description="Supports community groups monitoring drinking water quality in rural areas.",
        eligibility_text="Open to nonprofit organisations.",
        applicant_codes=["12"],
        award_floor=25_000,
        award_ceiling=150_000,
        close_date=TODAY + timedelta(days=90),
        jurisdiction="US",
        sectors=[Sector.ENVIRONMENT],
        cost_share_required=False,
        forms=[FormReference(name="SF-424", url="https://example.org/forms/sf424.pdf")],
        fetched_at=utcnow(),
    )
    defaults.update(overrides)
    return Opportunity(**defaults)


def applicant(**overrides) -> Applicant:
    defaults = dict(
        applicant_type=ApplicantType.NONPROFIT_501C3,
        country="US",
        sectors=[Sector.ENVIRONMENT],
        project_description="We test drinking water quality in rural wells and publish the results.",
        amount_sought=80_000,
    )
    defaults.update(overrides)
    return Applicant(**defaults)
