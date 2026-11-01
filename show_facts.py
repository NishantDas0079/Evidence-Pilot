"""Print what a trace shows: computed facts, tool calls and the final report.

Usage: python show_facts.py runs/two_bugs.s3.trace.jsonl
"""
import sys
from pathlib import Path

from agent.trace import Trace
from audit.evidence import derive_facts

if len(sys.argv) != 2:
    raise SystemExit("usage: python show_facts.py <trace file>")
records = Trace.load(Path(sys.argv[1]))
print("FACTS:\n" + derive_facts(records))
print("\nTOOL CALLS:")
for r in records:
    if r["kind"] == "tool_call":
        path = (r.get("args") or {}).get("path")
        print(f"  {r['id']} {r['name']}" + (f" {path}" if path else ""))
finals = [r for r in records if r["kind"] == "final_report"]
print("\nFINAL REPORT:\n" + (finals[-1]["text"] if finals else "(none)"))
