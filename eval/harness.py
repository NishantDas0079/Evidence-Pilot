import time
from concurrent.futures import ThreadPoolExecutor

from audit.service import audit_single
from eval.metrics import classification_report


def select(cases: list[dict], split: str) -> list[dict]:
    if split not in {"dev", "test", "all"}:
        raise ValueError(f"unknown split: {split}")
    return [c for c in cases if split == "all" or c["split"] == split]


def evaluate(cases: list[dict], records_by_trace: dict, llm, workers: int = 4) -> dict:
    def one(case: dict) -> dict:
        start = time.perf_counter()
        try:
            audit = audit_single(case["claim"], records_by_trace[case["trace"]], llm)
            pred, issues, n_ev = audit.status.value, audit.validation_issues, len(audit.evidence)
            explanation = audit.explanation
        except Exception as exc:  # an API failure counts as a wrong answer, never a crash
            pred, issues, n_ev = "ERROR", [f"{type(exc).__name__}: {str(exc)[:120]}"], 0
            explanation = ""
        return {
            "id": case["id"], "trace": case["trace"], "claim": case["claim"],
            "gold": case["label"], "pred": pred, "n_evidence": n_ev,
            "issues": issues, "explanation": explanation,
            "seconds": round(time.perf_counter() - start, 2),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, cases))

    gold = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    seconds = sorted(r["seconds"] for r in rows)
    return {
        "rows": rows,
        "metrics": classification_report(gold, pred),
        "false_reassurance": sum(r["gold"] != "SUPPORTED" and r["pred"] == "SUPPORTED" for r in rows),
        "downgraded_by_validator": sum(any("downgraded" in i for i in r["issues"]) for r in rows),
        "errors": sum(r["pred"] == "ERROR" for r in rows),
        "latency_median_s": seconds[len(seconds) // 2] if seconds else 0.0,
        "latency_max_s": seconds[-1] if seconds else 0.0,
    }
