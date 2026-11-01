import shutil
from pathlib import Path
 
import pytest
from fastapi.testclient import TestClient
 
from app import main, runs
from app.ratelimit import RateLimiter
from audit.llm import FakeJSONClient
 
SRC = Path(__file__).resolve().parent.parent / "eval" / "traces"
RUN = "synthetic_stale_pass"
 
 
def one_claim(user):
    return {"claims": [{"text": "All tests pass.", "verifiable": True}]}
 
 
def good_verdict(user):
    return {
        "status": "SUPPORTED",
        "explanation": "e",
        "citations": [{"item_id": "t001", "excerpt": "stats.py"}],
    }
 
 
def make_llm(verdict=good_verdict, extract=one_claim):
    return FakeJSONClient({"report_claims": extract, "claim_audit": verdict})
 
 
@pytest.fixture
def env(tmp_path, monkeypatch):
    d = tmp_path / "traces"
    shutil.copytree(SRC, d)
    monkeypatch.setenv("TRACE_DIR", str(d))
    state = {"llm": make_llm(), "limiter": RateLimiter(100, 100)}
    main.app.dependency_overrides[main.get_llm] = lambda: state["llm"]
    main.app.dependency_overrides[main.get_limiter] = lambda: state["limiter"]
    yield state
    main.app.dependency_overrides.clear()
 
 
@pytest.fixture
def client(env):
    return TestClient(main.app)
 
 
def post(client, report="All tests pass.", run_id=RUN):
    return client.post("/api/audit", json={"run_id": run_id, "report": report})
 
 
# --- pages and read-only endpoints ------------------------------------------------
 
def test_index_served_with_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and "EvidencePilot" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
 
 
def test_page_never_uses_innerhtml():
    html = (Path(main.__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    assert "innerHTML" not in html and "outerHTML" not in html and "insertAdjacentHTML" not in html
 
 
def test_health(client):
    assert client.get("/health").json()["status"] == "ok"
 
 
def test_list_runs(client):
    data = client.get("/api/runs").json()
    assert {r["id"] for r in data} >= {RUN, "synthetic_injection"}
    assert all(r["synthetic"] == r["id"].startswith("synthetic_") for r in data)
    flags = [r["synthetic"] for r in data]
    assert flags == sorted(flags)  # real runs are listed before synthetic ones
 
 
def test_get_run_has_items_and_facts(client):
    data = client.get(f"/api/runs/{RUN}").json()
    assert data["report"] and any(i["kind"] == "facts" for i in data["items"])
 
 
@pytest.mark.parametrize("bad", ["nope", "..%2F..%2Fetc%2Fpasswd", "a%00b"])
def test_unknown_or_traversal_run_is_404(client, bad):
    assert client.get(f"/api/runs/{bad}").status_code in (404, 422)
 
 
def test_load_run_rejects_traversal(env):
    assert runs.load_run("../traces/" + RUN) is None
    assert runs.load_run("") is None
 
 
# --- audit --------------------------------------------------------------------------
 
def test_audit_success(client):
    r = post(client)
    assert r.status_code == 200
    body = r.json()
    assert body["claims"][0]["status"] == "SUPPORTED"
    assert body["claims"][0]["evidence"][0]["excerpt"] == "stats.py"
 
 
def test_fabricated_quote_is_downgraded(client, env):
    env["llm"] = make_llm(verdict=lambda u: {
        "status": "SUPPORTED", "explanation": "e",
        "citations": [{"item_id": "t001", "excerpt": "this text is not in the record"}]})
    c = post(client).json()["claims"][0]
    assert c["status"] == "INSUFFICIENT_EVIDENCE" and c["validation_issues"]
 
 
def test_validation_errors(client):
    assert post(client, report="").status_code == 422
    assert post(client, report="x" * 4001).status_code == 422
    assert post(client, report="   ").status_code == 422
    assert client.post("/api/audit", json={"run_id": RUN}).status_code == 422
 
 
def test_unknown_run_audit_is_404(client):
    assert post(client, run_id="missing").status_code == 404
 
 
def test_malformed_extraction_is_502(client, env):
    env["llm"] = make_llm(extract=lambda u: {"nope": 1})
    assert post(client).status_code == 502
 
 
def test_unconfigured_model_is_503(client, env):
    def boom(u):
        raise RuntimeError("MODEL_API_KEY is not set")
    env["llm"] = make_llm(extract=boom)
    r = post(client)
    assert r.status_code == 503 and "MODEL_API_KEY" not in r.text
 
 
def test_rate_limit_returns_429(client, env):
    env["limiter"] = RateLimiter(1, 100)
    assert post(client).status_code == 200
    assert post(client).status_code == 429
 
 
def test_bad_requests_do_not_use_up_the_limit(client, env):
    env["limiter"] = RateLimiter(1, 100)
    assert post(client, run_id="missing").status_code == 404
    assert post(client, report="  ").status_code == 422
    assert post(client).status_code == 200
 
 
# --- limiter ------------------------------------------------------------------------
 
def test_limiter_per_ip_and_window():
    now = [0.0]
    lim = RateLimiter(2, 100, clock=lambda: now[0])
    assert lim.check("a") is None and lim.check("a") is None
    assert lim.check("a") is not None
    assert lim.check("b") is None
    now[0] = 3601
    assert lim.check("a") is None
 
 
def test_limiter_global_cap_and_reset():
    now = [0.0]
    lim = RateLimiter(10, 3, clock=lambda: now[0])
    assert [lim.check(f"ip{i}") for i in range(3)] == [None] * 3
    assert "daily" in lim.check("ip9")
    now[0] = 86401
    assert lim.check("ip9") is None
 
 
# --- model client -------------------------------------------------------------------
 
def test_meta_client_without_key_raises_runtime_error(monkeypatch):
    import dotenv
    from audit.llm import MetaJSONClient
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_API_KEY"):
        MetaJSONClient()