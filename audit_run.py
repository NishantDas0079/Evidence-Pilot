"""Audit an agent run.

Usage:
  python audit_run.py runs/average_bug.trace.jsonl
  python audit_run.py runs/average_bug.trace.jsonl --report-file my_report.txt
"""
import argparse
import json
from pathlib import Path

from agent.trace import Trace
from audit.llm import MetaJSONClient
from audit.service import audit_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--report-file", help="audit this text instead of the run's own final report")
    args = ap.parse_args()

    records = Trace.load(Path(args.trace))
    if args.report_file:
        report = Path(args.report_file).read_text(encoding="utf-8")
    else:
        finals = [r for r in records if r["kind"] == "final_report"]
        if not finals:
            raise SystemExit("trace has no final_report")
        report = finals[-1]["text"]

    result = audit_report(report, records, MetaJSONClient())

    print("REPORT UNDER AUDIT:\n" + result.report + "\n")
    print("FACTS COMPUTED FROM THE TRACE:\n" + result.facts + "\n")
    for i, c in enumerate(result.claims, 1):
        print(f"[{i}] {c.status.value}: {c.claim}")
        print(f"    {c.explanation}")
        for ev in c.evidence:
            print(f"    evidence {ev.item_id}: \"{ev.excerpt}\"")
        for issue in c.validation_issues:
            print(f"    check: {issue}")
    print("\nSUMMARY:", result.summary)

    out = Path(args.trace).with_suffix("").with_suffix(".audit.json")
    out.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
