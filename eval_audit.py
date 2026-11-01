"""Score the auditor's verdicts against hand-labeled claims.

  python eval_audit.py                 # dev split: use this while tuning
  python eval_audit.py --split test    # held-out split: run rarely, do NOT tune on its errors
"""
import argparse
import json
from pathlib import Path

from agent.trace import Trace
from audit.llm import MetaJSONClient
from eval.harness import evaluate, select
from eval.metrics import LABELS

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--effort", default=None, choices=["minimal", "low", "medium", "high"],
                    help="override the reasoning effort for every verdict call")
    args = ap.parse_args()

    cases = select(json.loads((ROOT / "eval" / "cases.json").read_text("utf-8"))["cases"], args.split)
    traces = sorted({c["trace"] for c in cases})
    missing = [t for t in traces if not (ROOT / t).exists()]
    if missing:
        raise SystemExit("missing trace files (copy them from runs/ into eval/traces/):\n  " + "\n  ".join(missing))
    records = {t: Trace.load(ROOT / t) for t in traces}

    if args.split != "dev":
        print("NOTE: held-out measurement. Do not change prompts based on these errors.\n")
    client = MetaJSONClient(effort=args.effort)
    result = evaluate(cases, records, client, workers=args.workers)

    m = result["metrics"]
    print(f"split={args.split}  claims={m['n']}  accuracy={m['accuracy']:.2f}  errors={result['errors']}")
    print(f"{'label':24}{'precision':>10}{'recall':>8}{'f1':>6}{'n':>4}")
    for label in LABELS:
        s = m["per_class"][label]
        print(f"{label:24}{s['precision']:>10.2f}{s['recall']:>8.2f}{s['f1']:>6.2f}{s['support']:>4}")
    print("\nconfusion (rows = gold, columns = predicted):")
    cols = LABELS + ["other"]
    print(f"{'':24}" + "".join(f"{c[:10]:>12}" for c in cols))
    for g in LABELS:
        print(f"{g:24}" + "".join(f"{m['confusion'][g][c]:>12}" for c in cols))
    print(f"\nfalse reassurance (gold not SUPPORTED, predicted SUPPORTED): {result['false_reassurance']}")
    print(f"verdicts downgraded by the citation validator: {result['downgraded_by_validator']}")
    print(f"latency per claim: median {result['latency_median_s']}s, max {result['latency_max_s']}s")

    u = client.usage
    cost = u["input_tokens"] * 1.25 / 1e6 + u["output_tokens"] * 4.25 / 1e6  # Standard-mode list prices
    print(f"effort={args.effort or 'default (medium)'}  tokens: in={u['input_tokens']} out={u['output_tokens']} "
          f"(reasoning {u['reasoning_tokens']})  est. cost ${cost:.4f} = ${cost / max(m['n'], 1):.5f} per claim")

    wrong = [r for r in result["rows"] if r["gold"] != r["pred"]]
    print(f"\nWRONG ({len(wrong)}):")
    for r in wrong:
        print(f"  {r['id']}: gold={r['gold']} pred={r['pred']}\n    claim: {r['claim']}\n    why: {r['explanation'][:160]}")

    out = ROOT / "eval" / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.split}.{args.effort or 'default'}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved to eval/results/{args.split}.{args.effort or 'default'}.json")


if __name__ == "__main__":
    main()
