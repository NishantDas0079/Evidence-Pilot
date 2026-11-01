from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, ValidationError

from audit.evidence import build_evidence, derive_facts
from audit.llm import JSONClient
from audit.prompts import (
    EXTRACT_NAME, EXTRACT_SCHEMA, EXTRACT_SYSTEM,
    VERDICT_NAME, VERDICT_SCHEMA, VERDICT_SYSTEM, build_verdict_prompt,
)
from audit.schemas import AuditReport, Citation, ClaimAudit, Status
from audit.validate import CitationValidator
from typing import Literal

MAX_REPORT_CHARS = 4000
VERDICT_EFFORT = "low"  # chosen on the dev split: same accuracy as medium/high, cheaper and faster
LIMITATIONS = [
    "Verdicts are judged by a language model and can be wrong. Every quote shown is checked to appear verbatim in the recorded trace, but that does not prove the quote supports the claim.",
    "Claims are checked only against the recorded trace. INSUFFICIENT_EVIDENCE means the trace does not settle the claim, not that the claim is false.",
]


class AuditError(Exception):
    pass


class _Claim(BaseModel):
    text: str
    verifiable: bool


class _Extraction(BaseModel):
    claims: list[_Claim]


class _Cite(BaseModel):
    item_id: str
    excerpt: str


class _Verdict(BaseModel):
    status: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    explanation: str
    citations: list[_Cite]


def _unusable(claim: str) -> ClaimAudit:
    return ClaimAudit(
        claim=claim,
        status=Status.INSUFFICIENT_EVIDENCE,
        explanation="The auditor returned unusable output, so no verdict was made.",
        validation_issues=["malformed model output"],
    )


def _audit_claim(claim: str, items, validator, llm: JSONClient) -> ClaimAudit:
    try:
        raw = llm.generate_json(
            system=VERDICT_SYSTEM,
            user=build_verdict_prompt(claim, items),
            schema_name=VERDICT_NAME,
            schema=VERDICT_SCHEMA,
            effort=VERDICT_EFFORT,
        )
        verdict = _Verdict.model_validate(raw)
    except ValidationError:
        return _unusable(claim)
    proposed = ClaimAudit(
        claim=claim,
        status=Status(verdict.status),
        explanation=verdict.explanation,
        evidence=[Citation(item_id=c.item_id, excerpt=c.excerpt) for c in verdict.citations],
    )
    return validator.validate(proposed)


def audit_report(
    report: str, records: list[dict], llm: JSONClient, max_workers: int = 4
) -> AuditReport:
    report = report.strip()
    if not report:
        raise ValueError("report is empty")
    if len(report) > MAX_REPORT_CHARS:
        raise ValueError(f"report longer than {MAX_REPORT_CHARS} characters")
    try:
        extraction = _Extraction.model_validate(
            llm.generate_json(
                system=EXTRACT_SYSTEM,
                user=f"<report>\n{report}\n</report>",
                schema_name=EXTRACT_NAME,
                schema=EXTRACT_SCHEMA,
                effort="low",
            )
        )
    except ValidationError as exc:
        raise AuditError("model returned a malformed claim list") from exc

    items = build_evidence(records)
    validator = CitationValidator(items)
    claims = [c for c in extraction.claims if c.text.strip()]

    def one(c: _Claim) -> ClaimAudit:
        if not c.verifiable:
            return ClaimAudit(
                claim=c.text,
                status=Status.NOT_VERIFIABLE,
                explanation="Not an independently checkable statement (opinion, prediction or vague).",
            )
        return _audit_claim(c.text, items, validator, llm)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        audits = list(pool.map(one, claims))  # map keeps claim order

    counts = Counter(a.status.value for a in audits)
    return AuditReport(
        report=report,
        claims=audits,
        summary={s.value: counts.get(s.value, 0) for s in Status},
        facts=derive_facts(records),
        limitations=LIMITATIONS,
    )


def audit_single(claim: str, records: list[dict], llm: JSONClient) -> ClaimAudit:
    """Verdict for one already-extracted claim (used to evaluate verdicts in isolation)."""
    items = build_evidence(records)
    return _audit_claim(claim, items, CitationValidator(items), llm)
