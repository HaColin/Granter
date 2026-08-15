"""FastAPI app: intake survey in, ranked shortlist out.

    uvicorn granter.app:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import intake, search, store
from .models import Confidence, Verdict
from .taxonomy import APPLICANT_TYPE_LABELS, SECTOR_LABELS

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Granter", description="Find grants you are actually eligible for.")

VERDICT_LABELS = {
    Verdict.ELIGIBLE: "Eligible",
    Verdict.LIKELY: "Likely eligible — verify",
    Verdict.NEAR_MISS: "Near miss",
    Verdict.INELIGIBLE: "Not eligible",
}

CONFIDENCE_LABELS = {
    Confidence.HIGH: "High confidence",
    Confidence.MEDIUM: "Medium confidence",
    Confidence.LOW: "Low confidence — source data incomplete",
}


def _base_context(request: Request) -> dict:
    return {
        "request": request,
        "verdict_labels": VERDICT_LABELS,
        "confidence_labels": CONFIDENCE_LABELS,
        "applicant_labels": APPLICANT_TYPE_LABELS,
        "sector_labels": SECTOR_LABELS,
        "free_access_note": search.FREE_ACCESS_NOTE,
    }


@app.get("/", response_class=HTMLResponse)
def survey(request: Request) -> HTMLResponse:
    corpus = store.load()
    context = _base_context(request)
    context.update(
        {
            # Rendered with every branch present; the browser reveals the ones
            # that apply, and to_applicant() discards answers to hidden ones.
            "questions": intake.QUESTIONS,
            "corpus": corpus,
        }
    )
    return templates.TemplateResponse(request, "survey.html", context)


@app.post("/results", response_class=HTMLResponse)
async def results(request: Request) -> HTMLResponse:
    form = await request.form()
    answers = {key: form.getlist(key) if key == "sectors" else form.get(key) for key in form.keys()}

    context = _base_context(request)

    try:
        applicant = intake.to_applicant(answers)
    except (KeyError, ValueError) as exc:
        context.update({"questions": intake.QUESTIONS, "corpus": store.load(), "error": str(exc)})
        return templates.TemplateResponse(request, "survey.html", context, status_code=400)

    corpus = store.load()
    result = search.run(applicant, corpus)

    context.update({"applicant": applicant, "result": result})
    return templates.TemplateResponse(request, "results.html", context)


@app.get("/health")
def health() -> dict:
    corpus = store.load()
    return {
        "status": "ok",
        "corpus_size": len(corpus),
        "sources": sorted(corpus.sources()),
        "fetched_at": corpus.fetched_at.isoformat() if corpus.fetched_at else None,
    }
