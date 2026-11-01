"""Turn an agent trace into numbered evidence items, plus facts computed by code.

The final report is NOT evidence: it is the thing being audited.
"""
import json
import re
from dataclasses import dataclass

MAX_ITEM_CHARS = 3000
FACTS_ID = "facts"
_COUNT_RE = re.compile(r"(\d+) (passed|failed|errors?|skipped)")


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    kind: str
    text: str


def _clip(text: str) -> str:
    return text if len(text) <= MAX_ITEM_CHARS else text[:MAX_ITEM_CHARS] + " [truncated]"


def _describe_call(record: dict) -> str:
    name, args = record["name"], record.get("args") or {}
    if name == "write_file":
        content = str(args.get("content", ""))
        return f"write_file path={args.get('path')} ({len(content)} chars)\n{content}"
    if name == "read_file":
        return f"read_file path={args.get('path')}"
    return name


def _num(record_id: str) -> int:
    return int(record_id[1:])


def _paired_calls(records: list[dict]):
    """Yield (call_record, ok) for each tool call, using the result that follows it."""
    for i, r in enumerate(records):
        if r["kind"] != "tool_call":
            continue
        nxt = records[i + 1] if i + 1 < len(records) else None
        ok = bool(nxt and nxt["kind"] == "tool_result" and nxt["name"] == r["name"] and nxt.get("ok", False))
        yield r, ok, (nxt if nxt and nxt["kind"] == "tool_result" else None)


def derive_facts(records: list[dict]) -> str:
    """Plain-language facts computed from the trace by code, not by a model.

    Only calls whose result succeeded count as reads or writes. Failed calls are listed apart.
    """
    pairs = list(_paired_calls(records))
    runs = [r for r in records if r["kind"] == "tool_result" and r["name"] == "run_tests"]
    writes = [c for c, ok, _ in pairs if c["name"] == "write_file" and ok]
    reads = [c for c, ok, _ in pairs if c["name"] == "read_file" and ok]
    failed = [(c, res) for c, ok, res in pairs if not ok]
    lines = [f"number of test runs: {len(runs)}"]
    if runs:
        last = runs[-1]
        code = re.search(r"exit_code:\s*(\S+)", last["text"])
        counts = {}
        for n, label in _COUNT_RE.findall(last["text"]):
            counts["error" if label.startswith("error") else label] = int(n)
        lines.append(
            f"last test run: record {last['id']}, exit_code {code.group(1) if code else 'unknown'}, "
            f"{counts.get('passed', 0)} passed, {counts.get('failed', 0)} failed, "
            f"{counts.get('error', 0)} errors"
        )
    else:
        lines.append("last test run: none")
    if writes:
        paths = sorted({str(w["args"].get("path")) for w in writes})
        lines.append(f"files written: {', '.join(paths)}")
        lines.append(f"last file write: record {writes[-1]['id']}")
    else:
        lines.append("files written: none")
    lines.append(
        "files read: " + (", ".join(sorted({str(r['args'].get('path')) for r in reads})) or "none")
    )
    if runs and writes:
        after = _num(runs[-1]["id"]) > _num(writes[-1]["id"])
        lines.append("tests were run after the last file write: " + ("yes" if after else "no"))
    lines.append(f"failed tool calls: {len(failed)}")
    for call, res in failed:
        target = (call.get("args") or {}).get("path")
        reason = (res["text"] if res else "no result recorded").splitlines()[0][:80]
        lines.append(f"failed call: {call['name']}{' ' + str(target) if target else ''} -> {reason}")
    return "\n".join(lines)


def build_evidence(records: list[dict]) -> list[EvidenceItem]:
    items = []
    for r in records:
        kind = r["kind"]
        if kind == "task":
            items.append(EvidenceItem(r["id"], "task", _clip(r["text"])))
        elif kind == "agent_message":
            items.append(EvidenceItem(r["id"], kind, _clip(r["text"])))
        elif kind == "tool_call":
            items.append(EvidenceItem(r["id"], kind, _clip(_describe_call(r))))
        elif kind == "tool_result":
            items.append(
                EvidenceItem(r["id"], kind, _clip(f"[{r['name']} ok={r['ok']}] {r['text']}"))
            )
    items.append(EvidenceItem(FACTS_ID, "facts", derive_facts(records)))
    return items
