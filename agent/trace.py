"""Append-only trace of everything the agent did. This is the audit evidence."""
import json
from pathlib import Path


class Trace:
    def __init__(self):
        self.records: list[dict] = []

    def add(self, kind: str, **fields) -> str:
        record_id = f"t{len(self.records) + 1:03d}"
        self.records.append({"id": record_id, "kind": kind, **fields})
        return record_id

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def load(path: Path) -> list[dict]:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
