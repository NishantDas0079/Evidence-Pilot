import logging
import os
from functools import lru_cache
from pathlib import Path

import openai
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import runs
from app.ratelimit import RateLimiter
from audit.llm import JSONClient, MetaJSONClient
from audit.schemas import AuditReport
from audit.service import MAX_REPORT_CHARS, AuditError, audit_report

logger = logging.getLogger("evidencepilot")
app = FastAPI(title="EvidencePilot", version="1.0.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"

CSP = (
    "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
    "base-uri 'none'; frame-ancestors 'none'"
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@lru_cache(maxsize=1)
def get_limiter() -> RateLimiter:
    return RateLimiter(
        per_ip_per_hour=int(os.getenv("AUDIT_PER_IP_PER_HOUR", "10")),
        global_per_day=int(os.getenv("AUDIT_GLOBAL_PER_DAY", "300")),
    )


@lru_cache(maxsize=1)
def get_llm() -> JSONClient:
    return MetaJSONClient(effort="low")


def client_ip(request: Request) -> str:
    if os.getenv("TRUST_PROXY") == "1":
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class AuditRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=80)
    report: str = Field(min_length=1, max_length=MAX_REPORT_CHARS)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "EvidencePilot"}


@app.get("/api/runs")
def list_runs():
    return runs.list_runs()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    loaded = runs.load_run(run_id)
    if loaded is None:
        raise HTTPException(404, "No such run.")
    summary, records = loaded
    return {**summary, "items": runs.run_items(records)}


@app.post("/api/audit", response_model=AuditReport)
def audit(
    body: AuditRequest,
    request: Request,
    llm: JSONClient = Depends(get_llm),
    limiter: RateLimiter = Depends(get_limiter),
):
    loaded = runs.load_run(body.run_id)
    if loaded is None:
        raise HTTPException(404, "No such run.")
    if not body.report.strip():
        raise HTTPException(422, "The report is empty.")
    refusal = limiter.check(client_ip(request))
    if refusal:
        raise HTTPException(429, refusal)
    _, records = loaded
    try:
        return audit_report(body.report, records, llm)
    except AuditError:
        raise HTTPException(502, "The model returned a malformed claim list. Please try again.")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except RuntimeError:
        logger.exception("Model client is not configured")
        raise HTTPException(503, "The model API is not configured on the server.")
    except openai.OpenAIError as exc:
        logger.exception("Model API call failed: %s", type(exc).__name__)
        raise HTTPException(502, "The model API request failed. Please try again.")
