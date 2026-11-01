import re

from audit.evidence import EvidenceItem
from audit.schemas import ClaimAudit, Status

NEEDS_EVIDENCE = {Status.SUPPORTED, Status.CONTRADICTED}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class CitationValidator:
    """A quote is only trusted if it appears verbatim in the cited trace item."""

    def __init__(self, items: list[EvidenceItem]):
        self._by_id = {i.id: i for i in items}

    def check(self, item_id: str, excerpt: str) -> list[str]:
        item = self._by_id.get(item_id)
        if item is None:
            return [f"unknown trace item: {item_id}"]
        if not excerpt.strip():
            return [f"empty excerpt for {item_id}"]
        if _norm(excerpt) not in _norm(item.text):
            return [f"excerpt not found in {item_id}"]
        return []

    def validate(self, audit: ClaimAudit) -> ClaimAudit:
        valid, issues = [], []
        for c in audit.evidence:
            problems = self.check(c.item_id, c.excerpt)
            if problems:
                issues.extend(problems)
            else:
                valid.append(c)
        status, explanation = audit.status, audit.explanation
        if status in NEEDS_EVIDENCE and not valid:
            status = Status.INSUFFICIENT_EVIDENCE
            explanation = (
                f"The auditor proposed {audit.status.value}, but none of its citations "
                "could be validated against the recorded trace, so no verdict is given."
            )
            issues.append(f"downgraded from {audit.status.value}: no valid citation")
        return audit.model_copy(
            update={
                "status": status,
                "explanation": explanation,
                "evidence": valid,
                "validation_issues": issues,
            }
        )
