"""Deterministic relevance scoring.

BM25 over the call text, plus a handful of explicit structural bonuses. No
embeddings and no model: every point in a score can be attributed to a term that
appears in both the applicant's project description and the retrieved call.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .models import Applicant, Match, Verdict
from .taxonomy import SECTOR_KEYWORDS, Sector

_TOKEN = re.compile(r"[a-z][a-z0-9\-]{2,}")

# Words that carry no signal in grant text -- they appear in nearly every call.
_STOPWORDS = frozenset(
    """the and for that with this from are will not you your our their its his her they
    them has have had was were been being which who whom whose what when where why how
    all any both each few more most other some such only own same than too very can just
    should now also may must shall into over under between during before after above
    below through about against because while under within without upon per via
    application applicant applicants award awards grant grants funding fund funds
    program programs project projects proposal proposals opportunity opportunities
    federal agency department office notice announcement eligible eligibility
    """.split()
)

BM25_K1 = 1.5
BM25_B = 0.75

# Structural bonuses, applied after the text score is normalised to 0..1.
SECTOR_BONUS = 0.35
AMOUNT_FIT_BONUS = 0.15
DEADLINE_FEASIBLE_BONUS = 0.10
CONFIDENCE_PENALTY = {"high": 0.0, "medium": -0.05, "low": -0.15}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _call_text(match: Match) -> str:
    opp = match.opportunity
    return " ".join([opp.title, opp.description, opp.eligibility_text])


def infer_sectors(text: str) -> list[Sector]:
    """Detect sectors in free call text. Ranking signal only."""
    low = text.lower()
    found = [s for s, terms in SECTOR_KEYWORDS.items() if any(t in low for t in terms)]
    return found or [Sector.OTHER]


class _Bm25Index:
    def __init__(self, documents: list[list[str]]) -> None:
        self.docs = documents
        self.n = len(documents)
        self.lengths = [len(d) for d in documents]
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.freqs = [Counter(d) for d in documents]

        df: Counter[str] = Counter()
        for doc in documents:
            df.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def score(self, index: int, query: list[str]) -> float:
        if not self.n or not self.avg_len:
            return 0.0
        freq = self.freqs[index]
        length = self.lengths[index]
        total = 0.0
        for term in query:
            tf = freq.get(term, 0)
            if not tf:
                continue
            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * length / self.avg_len)
            total += self.idf.get(term, 0.0) * (tf * (BM25_K1 + 1)) / denom
        return total


def score_matches(applicant: Applicant, matches: list[Match]) -> list[Match]:
    """Attach a 0..1-ish score to each match, in place, and return them sorted."""
    if not matches:
        return matches

    index = _Bm25Index([tokenize(_call_text(m)) for m in matches])
    query = tokenize(applicant.project_description)

    raw = [index.score(i, query) for i in range(len(matches))]
    ceiling = max(raw) or 1.0

    applicant_sectors = set(applicant.sectors)
    for match, raw_score in zip(matches, raw):
        score = raw_score / ceiling
        opp = match.opportunity

        sectors = set(opp.sectors) or set(infer_sectors(_call_text(match)))
        if applicant_sectors & sectors:
            score += SECTOR_BONUS

        if applicant.amount_sought is not None:
            floor = opp.award_floor if opp.award_floor is not None else 0
            ceil_ = opp.award_ceiling
            if floor <= applicant.amount_sought and (
                ceil_ is None or applicant.amount_sought <= ceil_
            ):
                score += AMOUNT_FIT_BONUS

        remaining = opp.days_remaining()
        if opp.rolling or (remaining is not None and remaining >= 30):
            score += DEADLINE_FEASIBLE_BONUS

        score += CONFIDENCE_PENALTY[match.confidence.value]
        match.score = round(max(score, 0.0), 4)

    order = {Verdict.ELIGIBLE: 0, Verdict.LIKELY: 1, Verdict.NEAR_MISS: 2, Verdict.INELIGIBLE: 3}
    matches.sort(key=lambda m: (order[m.verdict], -m.score))
    return matches
