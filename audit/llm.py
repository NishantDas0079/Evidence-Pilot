"""JSON-producing model clients: the real Meta one and a scripted fake for tests."""
import json
import os
from typing import Any, Callable, Protocol


class JSONClient(Protocol):
    def generate_json(
        self, *, system: str, user: str, schema_name: str, schema: dict[str, Any],
        effort: str = "medium",
    ) -> dict[str, Any]: ...


class FakeJSONClient:
    """handlers[schema_name](user_prompt) -> dict. Thread-safe enough for tests."""

    def __init__(self, handlers: dict[str, Callable[[str], dict]]):
        self.handlers = handlers
        self.calls: list[dict] = []

    def generate_json(self, *, system, user, schema_name, schema, effort="medium"):
        self.calls.append({"schema_name": schema_name, "system": system, "user": user, "effort": effort})
        return self.handlers[schema_name](user)


class MetaJSONClient:
    """Schema-constrained JSON via the Responses API (text.format verified Oct 27)."""

    def __init__(self, effort: str | None = None):
        import threading

        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        api_key = os.getenv("MODEL_API_KEY")
        if not api_key:
            raise RuntimeError("MODEL_API_KEY is not set")
        self._client = OpenAI(base_url="https://api.meta.ai/v1", api_key=api_key, timeout=120)
        self._model = os.getenv("MODEL_NAME", "muse-spark-1.3")
        self._effort = effort  # when set, overrides the per-call reasoning effort
        self._lock = threading.Lock()
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    def generate_json(self, *, system, user, schema_name, schema, effort="medium"):
        response = self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            reasoning={"effort": self._effort or effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            details = getattr(usage, "output_tokens_details", None)
            with self._lock:
                self.usage["calls"] += 1
                self.usage["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
                self.usage["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
                self.usage["reasoning_tokens"] += getattr(details, "reasoning_tokens", 0) or 0
        try:
            parsed = json.loads(response.output_text)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
