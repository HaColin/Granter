"""Applicant-type taxonomy and the mapping to Grants.gov eligibility codes.

The numeric codes are Grants.gov's published "Eligible Applicants" taxonomy and
appear verbatim in the ``applicantTypes`` field of the search/fetch API. They are
kept in one place so that a change in the upstream taxonomy is a one-file edit.
"""

from __future__ import annotations

from enum import Enum

# --- Grants.gov eligible-applicant codes -----------------------------------

GRANTS_GOV_APPLICANT_CODES: dict[str, str] = {
    "00": "State governments",
    "01": "County governments",
    "02": "City or township governments",
    "04": "Special district governments",
    "05": "Independent school districts",
    "06": "Public and State controlled institutions of higher education",
    "07": "Native American tribal governments (Federally recognized)",
    "08": "Public housing authorities / Indian housing authorities",
    "11": "Native American tribal organizations (other than Federally recognized)",
    "12": "Nonprofits with 501(c)(3) status, other than institutions of higher education",
    "13": "Nonprofits without 501(c)(3) status, other than institutions of higher education",
    "20": "Private institutions of higher education",
    "21": "Individuals",
    "22": "For-profit organizations other than small businesses",
    "23": "Small businesses",
    "25": "Others (see text field entitled 'Additional Information on Eligibility')",
    "99": "Unrestricted",
}

#: Codes that do not name a specific applicant class. A match on one of these is
#: never treated as proof of eligibility -- only as "read the call text".
AMBIGUOUS_CODES: frozenset[str] = frozenset({"25", "99"})


class ApplicantType(str, Enum):
    """What the applicant legally *is*. The hardest eligibility filter."""

    INDIVIDUAL = "individual"
    INFORMAL_GROUP = "informal_group"
    NONPROFIT_501C3 = "nonprofit_501c3"
    NONPROFIT_OTHER = "nonprofit_other"
    FOR_PROFIT_SMALL = "for_profit_small"
    FOR_PROFIT_OTHER = "for_profit_other"
    ACADEMIC_PUBLIC = "academic_public"
    ACADEMIC_PRIVATE = "academic_private"
    GOVERNMENT_STATE = "government_state"
    GOVERNMENT_LOCAL = "government_local"
    TRIBAL_GOVERNMENT = "tribal_government"
    TRIBAL_ORGANIZATION = "tribal_organization"
    SCHOOL_DISTRICT = "school_district"
    HOUSING_AUTHORITY = "housing_authority"


APPLICANT_TYPE_LABELS: dict[ApplicantType, str] = {
    ApplicantType.INDIVIDUAL: "Individual (no legal entity)",
    ApplicantType.INFORMAL_GROUP: "Informal / unincorporated group",
    ApplicantType.NONPROFIT_501C3: "Nonprofit with 501(c)(3) status",
    ApplicantType.NONPROFIT_OTHER: "Nonprofit without 501(c)(3) status",
    ApplicantType.FOR_PROFIT_SMALL: "Small business",
    ApplicantType.FOR_PROFIT_OTHER: "For-profit organization (not a small business)",
    ApplicantType.ACADEMIC_PUBLIC: "Public / state-controlled institution of higher education",
    ApplicantType.ACADEMIC_PRIVATE: "Private institution of higher education",
    ApplicantType.GOVERNMENT_STATE: "State government",
    ApplicantType.GOVERNMENT_LOCAL: "County, city or township government",
    ApplicantType.TRIBAL_GOVERNMENT: "Native American tribal government (federally recognized)",
    ApplicantType.TRIBAL_ORGANIZATION: "Native American tribal organization (other)",
    ApplicantType.SCHOOL_DISTRICT: "Independent school district",
    ApplicantType.HOUSING_AUTHORITY: "Public housing authority",
}

#: Which Grants.gov codes an applicant of each type can satisfy.
#: Deliberately conservative: a type is listed only where the mapping is exact.
APPLICANT_TYPE_TO_CODES: dict[ApplicantType, frozenset[str]] = {
    ApplicantType.INDIVIDUAL: frozenset({"21"}),
    # An unincorporated group is not a legal person, so the group cannot hold an
    # award. It maps to "Individuals" because the route that exists is for one
    # member to be the named applicant -- see the caution in eligibility.py.
    ApplicantType.INFORMAL_GROUP: frozenset({"21"}),
    ApplicantType.NONPROFIT_501C3: frozenset({"12"}),
    ApplicantType.NONPROFIT_OTHER: frozenset({"13"}),
    ApplicantType.FOR_PROFIT_SMALL: frozenset({"23"}),
    ApplicantType.FOR_PROFIT_OTHER: frozenset({"22"}),
    ApplicantType.ACADEMIC_PUBLIC: frozenset({"06"}),
    ApplicantType.ACADEMIC_PRIVATE: frozenset({"20"}),
    ApplicantType.GOVERNMENT_STATE: frozenset({"00"}),
    ApplicantType.GOVERNMENT_LOCAL: frozenset({"01", "02", "04"}),
    ApplicantType.TRIBAL_GOVERNMENT: frozenset({"07"}),
    ApplicantType.TRIBAL_ORGANIZATION: frozenset({"11"}),
    ApplicantType.SCHOOL_DISTRICT: frozenset({"05"}),
    ApplicantType.HOUSING_AUTHORITY: frozenset({"08"}),
}

#: Types that have no separate legal existence from the person behind them.
#: These get the fiscal-sponsorship conversation instead of a grant list.
NO_LEGAL_ENTITY_TYPES: frozenset[ApplicantType] = frozenset(
    {ApplicantType.INDIVIDUAL, ApplicantType.INFORMAL_GROUP}
)


class Sector(str, Enum):
    HEALTH = "health"
    EDUCATION = "education"
    ENVIRONMENT = "environment"
    ARTS = "arts"
    RESEARCH = "research"
    INFRASTRUCTURE = "infrastructure"
    HOUSING = "housing"
    FOOD_AGRICULTURE = "food_agriculture"
    COMMUNITY_DEVELOPMENT = "community_development"
    HUMAN_RIGHTS = "human_rights"
    TECHNOLOGY = "technology"
    OTHER = "other"


SECTOR_LABELS: dict[Sector, str] = {
    Sector.HEALTH: "Health",
    Sector.EDUCATION: "Education",
    Sector.ENVIRONMENT: "Environment / climate",
    Sector.ARTS: "Arts & culture",
    Sector.RESEARCH: "Scientific research",
    Sector.INFRASTRUCTURE: "Infrastructure",
    Sector.HOUSING: "Housing",
    Sector.FOOD_AGRICULTURE: "Food & agriculture",
    Sector.COMMUNITY_DEVELOPMENT: "Community & economic development",
    Sector.HUMAN_RIGHTS: "Human rights & justice",
    Sector.TECHNOLOGY: "Technology",
    Sector.OTHER: "Other",
}

#: Terms used to detect a sector in free call text when a funder publishes no
#: structured category. Used for ranking only -- never for hard eligibility.
SECTOR_KEYWORDS: dict[Sector, tuple[str, ...]] = {
    Sector.HEALTH: ("health", "medical", "clinical", "disease", "patient", "mental health"),
    Sector.EDUCATION: ("education", "school", "student", "teacher", "curriculum", "literacy"),
    Sector.ENVIRONMENT: ("environment", "climate", "conservation", "ecosystem", "emissions", "wildlife"),
    Sector.ARTS: ("arts", "artist", "museum", "humanities", "cultural", "heritage"),
    Sector.RESEARCH: ("research", "investigator", "laboratory", "scientific", "study", "fellowship"),
    Sector.INFRASTRUCTURE: ("infrastructure", "transportation", "transit", "bridge", "broadband", "water system"),
    Sector.HOUSING: ("housing", "homeless", "shelter", "rental", "tenant"),
    Sector.FOOD_AGRICULTURE: ("agriculture", "farm", "food", "nutrition", "rural"),
    Sector.COMMUNITY_DEVELOPMENT: ("community", "economic development", "workforce", "small business", "neighborhood"),
    Sector.HUMAN_RIGHTS: ("justice", "civil rights", "equity", "legal aid", "immigrant", "victim"),
    Sector.TECHNOLOGY: ("technology", "software", "cyber", "data", "artificial intelligence", "computing"),
}
