"""Run the agent on a task with the real Meta model and save its trace.

Usage:
  python run_agent.py average_bug
  python run_agent.py average_bug --max-steps 3     # stop early: a real but unfinished run
"""
import argparse
from pathlib import Path

from agent.runner import MAX_STEPS, make_meta_respond, run_agent

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    task_dir = ROOT / "tasks" / args.task
    task = (task_dir / "task.txt").read_text(encoding="utf-8").strip()
    steps = args.max_steps or MAX_STEPS
    result = run_agent(task, task_dir / "files", make_meta_respond(), max_steps=steps)

    suffix = f".s{args.max_steps}" if args.max_steps else ""
    out = ROOT / "runs" / f"{args.task}{suffix}.trace.jsonl"
    result.trace.save(out)

    print(f"finished={result.finished} steps={result.steps}")
    for r in result.trace.records:
        if r["kind"] == "tool_call":
            print(f"  {r['id']} tool_call {r['name']}")
        elif r["kind"] == "tool_result":
            first = r["text"].splitlines()[0] if r["text"] else ""
            print(f"  {r['id']} tool_result ok={r['ok']} {first[:60]}")
    print("\nFINAL REPORT:\n" + result.final_report)
    print(f"\ntrace saved to {out}")


if __name__ == "__main__":
    main()
