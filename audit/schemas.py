from enum import Enum

from pydantic import BaseModel, Field


class Status(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_VERIFIABLE = "NOT_VERIFIABLE"


class Citation(BaseModel):
    item_id: str
    excerpt: str


class ClaimAudit(BaseModel):
    claim: str
    status: Status
    explanation: str
    evidence: list[Citation] = Field(default_factory=list)
    validation_issues: list[str] = Field(default_factory=list)


class AuditReport(BaseModel):
    report: str
    claims: list[ClaimAudit]
    summary: dict[str, int]
    facts: str
    limitations: list[str]
