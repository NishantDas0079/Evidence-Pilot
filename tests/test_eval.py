import json
from pathlib import Path

import pytest

from agent.runner import run_agent
from audit.llm import FakeJSONClient
from eval.harness import evaluate, select
from eval.metrics import LABELS, classification_report

ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "tasks" / "average_bug" / "files"
TASK = (ROOT / "tasks" / "average_bug" / "task.txt").read_text(encoding="utf-8").strip()
FIXED = (FILES / "stats.py").read_text(encoding="utf-8").replace("(len(numbers) + 1)", "len(numbers)")


def call(name, args, cid):
    return {"type": "function_call", "call_id": cid, "name": name, "arguments": json.dumps(args)}


@pytest.fixture(scope="module")
def records():
    turns = [
        [call("write_file", {"path": "stats.py", "content": FIXED}, "a")],
        [call("run_tests", {}, "b")],
        [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
    ]
    return run_agent(TASK, FILES, lambda h: turns.pop(0)).trace.records


def verdict(status, item="facts", excerpt="exit_code 0"):
    return {"status": status, "explanation": "e", "citations": [{"item_id": item, "excerpt": excerpt}]}


def case(i, claim, label, split="dev"):
    return {"id": i, "trace": "t", "split": split, "claim": claim, "label": label}


# --- metrics ---------------------------------------------------------------------

def test_metrics_known_values():
    gold = ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    pred = ["SUPPORTED", "CONTRADICTED", "CONTRADICTED", "SUPPORTED"]
    m = classification_report(gold, pred)
    s, c, i = (m["per_class"][k] for k in LABELS)
    assert (s["precision"], s["recall"], s["support"]) == (0.5, 0.5, 2)
    assert c["precision"] == 0.5 and c["recall"] == 1.0
    assert (i["precision"], i["recall"], i["f1"]) == (0.0, 0.0, 0.0)
    assert m["accuracy"] == 0.5
    assert m["confusion"]["SUPPORTED"] == {"SUPPORTED": 1, "CONTRADICTED": 1, "INSUFFICIENT_EVIDENCE": 0, "other": 0}


def test_error_predictions_count_as_wrong_and_land_in_other():
    m = classification_report(["SUPPORTED", "CONTRADICTED"], ["ERROR", "CONTRADICTED"])
    assert m["accuracy"] == 0.5
    assert m["confusion"]["SUPPORTED"]["other"] == 1
    assert m["per_class"]["SUPPORTED"]["recall"] == 0.0


def test_metrics_length_mismatch():
    with pytest.raises(ValueError):
        classification_report(["SUPPORTED"], [])


# --- harness -----------------------------------------------------------------------

def run_eval(records, cases, table, workers=3):
    def handler(user):
        claim = user.split("<claim>")[1].split("</claim>")[0]
        out = table[claim]
        if isinstance(out, Exception):
            raise out
        return out
    return evaluate(cases, {"t": records}, FakeJSONClient({"claim_audit": handler}), workers=workers)


def test_harness_scores_and_counts_false_reassurance(records):
    cases = [
        case("1", "right", "SUPPORTED"),
        case("2", "falsely reassured", "CONTRADICTED"),
        case("3", "contradicted ok", "CONTRADICTED"),
    ]
    table = {
        "right": verdict("SUPPORTED"),
        "falsely reassured": verdict("SUPPORTED"),
        "contradicted ok": verdict("CONTRADICTED"),
    }
    result = run_eval(records, cases, table)
    assert [r["id"] for r in result["rows"]] == ["1", "2", "3"]
    assert result["false_reassurance"] == 1
    assert result["metrics"]["accuracy"] == pytest.approx(2 / 3)


def test_harness_counts_validator_downgrades(records):
    cases = [case("1", "fabricated", "SUPPORTED")]
    table = {"fabricated": verdict("SUPPORTED", excerpt="99 passed")}
    result = run_eval(records, cases, table)
    assert result["rows"][0]["pred"] == "INSUFFICIENT_EVIDENCE"
    assert result["downgraded_by_validator"] == 1


def test_api_failure_is_an_error_row_not_a_crash(records):
    cases = [case("1", "boom", "SUPPORTED"), case("2", "fine", "SUPPORTED")]
    table = {"boom": RuntimeError("503 from model"), "fine": verdict("SUPPORTED")}
    result = run_eval(records, cases, table)
    assert result["errors"] == 1
    assert result["rows"][0]["pred"] == "ERROR" and "RuntimeError" in result["rows"][0]["issues"][0]
    assert result["rows"][1]["pred"] == "SUPPORTED"


def test_select_splits():
    cases = [case("1", "a", "SUPPORTED", "dev"), case("2", "b", "SUPPORTED", "test")]
    assert [c["id"] for c in select(cases, "dev")] == ["1"]
    assert [c["id"] for c in select(cases, "test")] == ["2"]
    assert len(select(cases, "all")) == 2
    with pytest.raises(ValueError):
        select(cases, "oops")


# --- the hand-labeled gold file ----------------------------------------------------------

def test_gold_cases_are_well_formed():
    cases = json.loads((ROOT / "eval" / "cases.json").read_text(encoding="utf-8"))["cases"]
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        assert c["label"] in LABELS and c["split"] in {"dev", "test"}
        assert c["claim"].strip() and c["trace"].startswith("eval/traces/")


def test_no_trace_is_shared_between_dev_and_test():
    cases = json.loads((ROOT / "eval" / "cases.json").read_text(encoding="utf-8"))["cases"]
    dev = {c["trace"] for c in cases if c["split"] == "dev"}
    test = {c["trace"] for c in cases if c["split"] == "test"}
    assert dev and test and not (dev & test)


def test_verdict_prompt_states_the_action_rubric():
    from audit.prompts import VERDICT_SYSTEM
    assert "complete record of every action" in VERDICT_SYSTEM
    assert "INSUFFICIENT_EVIDENCE is for claims that go beyond" in VERDICT_SYSTEM
    assert "judged by that run's own result" in VERDICT_SYSTEM


# --- the shipped synthetic traces --------------------------------------------------

SYNTHETIC = {
    "synthetic_stale_pass": "tests were run after the last file write: no",
    "synthetic_failing_overclaim": "exit_code 1, 3 passed, 2 failed",
    "synthetic_injection": "NOTE TO AUDITOR",
    "synthetic_partial_fix": "exit_code 1, 4 passed, 2 failed",
    "synthetic_tool_error": "failed call: read_file config.yaml",
    "synthetic_compound": "exit_code 0, 6 passed",
}


@pytest.mark.parametrize("name,expected", SYNTHETIC.items())
def test_synthetic_traces_exist_and_show_their_scenario(name, expected):
    from agent.trace import Trace
    from audit.evidence import build_evidence

    recs = Trace.load(ROOT / "eval" / "traces" / f"{name}.trace.jsonl")
    shown = "\n".join(i.text for i in build_evidence(recs))
    assert expected in shown


def test_every_synthetic_case_points_at_a_shipped_trace():
    cases = json.loads((ROOT / "eval" / "cases.json").read_text(encoding="utf-8"))["cases"]
    for c in cases:
        if "/synthetic_" in c["trace"]:
            assert (ROOT / c["trace"]).exists(), c["trace"]
