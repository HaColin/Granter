# Granter

Finding grants is fragmented and opaque. Granter takes a structured intake survey and
returns a ranked list of calls you are plausibly eligible for — each with the official
source link, the deadline, why you match, what would disqualify you, and the forms the
application requires.

This is the first build: **the intake survey and the eligibility engine**, with a live
Grants.gov connector behind them. Matching is deterministic — structured filters plus
BM25 over the retrieved call text. No model is in the loop, so nothing in a result can
be hallucinated.

## The rule everything else follows

**Granter never invents a grant, deadline, amount, or URL.** The repository ships with an
empty corpus. Every record comes from an ingest run against a live source and carries the
URL it came from and the timestamp it was retrieved. Where a funder did not publish a
field, the record stores `None`, the field name goes into `missing_fields`, and the UI
says "not published" rather than showing a number nobody stated.

The synthetic opportunities in `tests/fixtures.py` are invented, are labelled as such,
and are never loaded by the application.

## Running it

On Windows, double-click **Start Granter.bat**. It installs what it needs, fetches the
opportunities if there are none yet, and opens the browser. Everything below is what it
does for you.

```bash
python -m granter.launch          # the same thing, from a terminal
```

Or step by step:

```bash
pip install -r requirements.txt
```

Fetch live opportunities (needs outbound access to `api.grants.gov`):

```bash
python -m granter.ingest --limit 200
```

Start the app:

```bash
python -m uvicorn granter.app:app --reload
```

Then open http://localhost:8000. With an empty corpus the app says so and refuses to
show results rather than filling the page with something it did not retrieve.

## Two ways in

The form at `/` is the primary path and needs no API key.

`/chat` is a conversational alternative: describe the project in plain language and a
Google Gemini model fills in the same survey, which you review before searching. Enable
it with a key:

```bash
export GEMINI_API_KEY=...        # or GOOGLE_API_KEY
python -m granter.chat           # verify the key and the response shape
```

`python -m granter.chat` makes one real call and reports exactly what worked or failed —
the first live call is the one thing the test suite cannot cover. If the default model
has been renamed, it tries alternates and tells you which to use.

Without a key, `/chat` says so and points at the form. The form is never gated behind
the model.

**The model fills in the form and does nothing else.** It has no access to grant data,
never names a grant or a deadline, and never decides eligibility — the structured
response schema it is constrained to contains exactly the survey's own fields and
nowhere for an opportunity to be returned. Its output is parsed into an `Applicant` and
handed to the same deterministic engine the form feeds, and the required fields are
checked here rather than trusted from the model. The worst a hallucination can do is
mis-fill a survey field you can see and correct.

Run the tests:

```bash
python -m pytest -q
```

## How a verdict is reached

`granter/eligibility.py` runs each call through the same checks and returns the note that
explains every one of them. Notes come in four kinds:

| Kind | Meaning |
| --- | --- |
| `match` | A stated applicant answer satisfies a stated field on the call. |
| `caution` | Eligible, but something needs verifying or preparing first. |
| `blocker` | The call, as retrieved, excludes this applicant. |
| `unknown` | The source did not publish the field. Never treated as a pass. |

Those roll up into a verdict:

* **Eligible** — the call names your applicant type and nothing is unknown.
* **Likely eligible** — matched, but through an ambiguous code (`Others`, `Unrestricted`)
  or with fields the source left blank. Verify before investing time.
* **Near miss** — your applicant type fits, but something else does not (budget above the
  ceiling, SAM.gov registration that cannot complete before the deadline). Shown in a
  separate list, labelled, never mixed in with real matches.
* **Also eligible, but unrelated** — you qualify, but the call's text has almost nothing
  in common with your project. Counted, with the top few named, rather than listed in
  full. On a 2,484-record corpus this is the difference between a 32-row shortlist and a
  1,607-row one. They are not discarded: keyword scoring is not certain enough to hide
  an eligible call outright.
* **Not eligible** — the call excludes your type, or the deadline has passed. Not shown.

Confidence is separate from verdict: it drops to `low` whenever the source document left
a field blank, and to `medium` when the record is more than a week old.

## Individuals

Most government grants go to organisations, and US federal grants are almost never
available to individuals for personal needs. Granter says this before it shows anything,
rather than returning a list of calls the person cannot receive:

* An **individual or informal group** gets a blocking advisory explaining the constraint,
  followed by the routes that do work — fiscal sponsorship, scholarships and fellowships,
  or applying jointly with a partner organisation. Only calls that explicitly name
  individuals (Grants.gov code `21`) can then match.
* With a **fiscal sponsor**, the sponsor is the legal applicant, so eligibility is
  evaluated against the sponsoring 501(c)(3) — and the UI says that is what happened.

## Sources

Built on the machine-readable ones first:

* **Grants.gov** — implemented (`granter/sources/grants_gov.py`), via the public
  Search2 / FetchOpportunity API. Federal money only.
* **California Grants Portal** — implemented (`granter/sources/ca_grants.py`), via the
  state's CKAN open-data publication on data.ca.gov, refreshed daily. State money, so
  no SAM.gov or UEI prerequisite — usually the more realistic pool for a small
  California applicant.
* **EU Funding & Tenders Portal** — implemented (`granter/sources/eu_portal.py`), via
  the portal's own bulk reference file. Covers Horizon Europe, LIFE, Erasmus+, CEF,
  Digital Europe and the rest. The file is ~128 MB and cached on disk for a day.
  Tenders are excluded: a procurement contract is not a grant.
* CORDIS, NIH Guide, UKRI Funding Finder, World Bank — next, same `Opportunity` shape.

The EU source publishes **no award amounts and no descriptive prose**, so those records
carry `None` budgets with the field names in `missing_fields`, and rank on thinner text
than a Grants.gov record. That is a limitation of the source, recorded rather than
filled in with a guess.

Fetch one source or all of them:

```bash
python -m granter.ingest --source all --limit 2000 --include-forecasted --replace
python -m granter.ingest --source ca_grants --limit 300
```

That pulls everything both sources publish — about 1,850 records — in well under a
minute. Each federal opportunity needs its own detail request, so those run eight at a
time; the concurrency is deliberately modest because this is a free public API.

A source that is unreachable is reported and skipped; the others still land, and
`--replace` only discards records from sources this run actually refetched.

Two kinds of source are **listed, never scraped**, and appear on the results page under
"Not searched — check these yourself":

* Subscription indexes: Candid, Devex, GrantStation, Pivot-RP, Research Professional.
* Public funders with no machine-readable feed: the African Development Bank, the Asian
  and Inter-American Development Banks, the UN Partner Portal, GlobalGiving.

The second group is free to browse — the only barrier is that it has to be done by hand.
Naming them is how a user can tell what a search did *not* cover. There is currently no
open feed for African development funding that could be ingested honestly.

## Layout

```
granter/
  taxonomy.py           applicant types, Grants.gov eligibility codes, sectors
  models.py             Opportunity, Applicant, Match — provenance on every record
  intake.py             the 11-field survey, with branching predicates
  eligibility.py        the deterministic rules engine
  ranking.py            BM25 + structural bonuses
  search.py             orchestration, near-miss split, paywalled-source referrals
  store.py              the JSON corpus (ships empty)
  ingest.py             CLI: python -m granter.ingest
  sources/grants_gov.py federal connector
  sources/ca_grants.py   California state connector
  sources/eu_portal.py   EU Funding & Tenders connector
  chat.py               conversational intake via Gemini (optional)
  app.py, templates/    FastAPI + server-rendered HTML
tests/                  132 tests; fixtures are synthetic and clearly marked
```

## Verifying the connector after an upstream change

```bash
python -m granter.ingest --probe
```

Prints the raw payload keys the normaliser depends on, and the first normalised record.
If Grants.gov changes a field name, `normalise()` raises rather than emitting a record
with plausible-looking gaps.
