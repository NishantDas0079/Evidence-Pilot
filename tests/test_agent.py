import json
import os
from pathlib import Path

import pytest

from agent.runner import run_agent
from agent.trace import Trace
from agent.workspace import ToolError, Workspace

TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "average_bug"
FILES = TASK_DIR / "files"
TASK = (TASK_DIR / "task.txt").read_text(encoding="utf-8").strip()

FIXED = (FILES / "stats.py").read_text(encoding="utf-8").replace(
    "(len(numbers) + 1)", "len(numbers)"
)


def call(name, args, call_id):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(args),
    }


def message(text):
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


def scripted(*turns):
    """Fake model: returns one scripted turn per call and records histories."""
    queue = list(turns)
    seen = []

    def respond(history):
        seen.append(list(history))
        if not queue:
            raise AssertionError("fake model has no turns left")
        return queue.pop(0)

    respond.seen = seen
    return respond


@pytest.fixture
def ws():
    w = Workspace(FILES)
    yield w
    w.cleanup()


# --- workspace tools -------------------------------------------------------

def test_list_and_read(ws):
    assert "stats.py" in ws.list_files()
    assert "def average" in ws.read_file("stats.py")


@pytest.mark.parametrize("bad", ["../secret.txt", "/etc/passwd", "sub/../../x"])
def test_paths_outside_workspace_rejected(ws, bad):
    with pytest.raises(ToolError):
        ws.read_file(bad)
    with pytest.raises(ToolError):
        ws.write_file(bad, "x")


def test_write_then_read_roundtrip(ws):
    ws.write_file("new/notes.txt", "hello")
    assert ws.read_file("new/notes.txt") == "hello"


def test_unknown_tool_and_missing_file(ws):
    with pytest.raises(ToolError):
        ws.call("rm_rf", {})
    with pytest.raises(ToolError):
        ws.read_file("nope.py")


def test_tests_fail_before_fix_and_pass_after(ws):
    before = ws.run_tests()
    assert before.startswith("exit_code: 1")
    ws.write_file("stats.py", FIXED)
    after = ws.run_tests()
    assert after.startswith("exit_code: 0")


def test_test_runner_hides_api_key(ws, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "sk-should-not-leak")
    ws.write_file("stats.py", FIXED)  # so the only possible failure is the env check
    ws.write_file(
        "test_env.py",
        "import os\n\ndef test_no_key():\n    assert 'MODEL_API_KEY' not in os.environ\n",
    )
    out = ws.run_tests()
    assert out.startswith("exit_code: 0"), out
    assert "sk-should-not-leak" not in out


def test_infinite_test_is_stopped(ws, monkeypatch):
    monkeypatch.setattr("agent.workspace.TEST_TIMEOUT_S", 2)
    ws.write_file("test_hang.py", "import time\n\ndef test_hang():\n    time.sleep(60)\n")
    assert "timeout" in ws.run_tests()


# --- the loop ----------------------------------------------------------------

def test_full_run_traces_every_step():
    respond = scripted(
        [call("read_file", {"path": "stats.py"}, "c1")],
        [call("write_file", {"path": "stats.py", "content": FIXED}, "c2")],
        [call("run_tests", {}, "c3")],
        [message("Fixed the divisor. All 5 tests pass.")],
    )
    result = run_agent(TASK, FILES, respond)
    assert result.finished and result.steps == 4
    kinds = [r["kind"] for r in result.trace.records]
    assert kinds == [
        "task",
        "tool_call", "tool_result",
        "tool_call", "tool_result",
        "tool_call", "tool_result",
        "final_report",
    ]
    test_result = result.trace.records[6]
    assert test_result["name"] == "run_tests"
    assert test_result["text"].startswith("exit_code: 0")
    assert "pass" in result.final_report


def test_narration_with_tool_call_is_traced_once():
    respond = scripted(
        [message("Let me look at the files."), call("list_files", {}, "c1")],
        [message("All done.")],
    )
    records = run_agent(TASK, FILES, respond).trace.records
    texts = [r["text"] for r in records if r["kind"] in {"agent_message", "final_report"}]
    assert texts == ["Let me look at the files.", "All done."]


def test_trace_ids_are_unique_and_ordered():
    respond = scripted([call("list_files", {}, "c1")], [message("done")])
    ids = [r["id"] for r in run_agent(TASK, FILES, respond).trace.records]
    assert ids == sorted(ids) and len(ids) == len(set(ids))


def test_history_is_replayed_with_tool_outputs():
    respond = scripted([call("list_files", {}, "c1")], [message("done")])
    run_agent(TASK, FILES, respond)
    second_turn = respond.seen[1]
    assert any(i.get("type") == "function_call" for i in second_turn)
    outputs = [i for i in second_turn if i.get("type") == "function_call_output"]
    assert outputs and outputs[0]["call_id"] == "c1"


def test_parallel_calls_each_get_a_result():
    respond = scripted(
        [call("list_files", {}, "a"), call("read_file", {"path": "stats.py"}, "b")],
        [message("done")],
    )
    result = run_agent(TASK, FILES, respond)
    results = [r for r in result.trace.records if r["kind"] == "tool_result"]
    assert [r["name"] for r in results] == ["list_files", "read_file"]


def test_tool_errors_become_visible_results_not_crashes():
    respond = scripted(
        [call("read_file", {"path": "../../etc/passwd"}, "c1")],
        [message("I could not read that file.")],
    )
    result = run_agent(TASK, FILES, respond)
    err = next(r for r in result.trace.records if r["kind"] == "tool_result")
    assert err["ok"] is False and "outside the workspace" in err["text"]
    assert result.finished


def test_bad_json_arguments_do_not_crash():
    bad = {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": "{not json"}
    respond = scripted([bad], [message("done")])
    result = run_agent(TASK, FILES, respond)
    err = next(r for r in result.trace.records if r["kind"] == "tool_result")
    assert err["ok"] is False and "not valid JSON" in err["text"]


def test_step_limit_marks_run_incomplete():
    turns = [[call("list_files", {}, f"c{i}")] for i in range(3)]
    result = run_agent(TASK, FILES, scripted(*turns), max_steps=3)
    assert result.finished is False
    last = result.trace.records[-1]
    assert last["kind"] == "final_report" and last.get("incomplete") is True


def test_workspace_removed_after_run():
    respond = scripted([message("nothing to do")])
    created = []
    original = Workspace.__init__

    def spy(self, source_dir):
        original(self, source_dir)
        created.append(self.root)

    Workspace.__init__ = spy
    try:
        run_agent(TASK, FILES, respond)
    finally:
        Workspace.__init__ = original
    assert created and not created[0].exists()


def test_source_task_files_are_never_modified():
    before = (FILES / "stats.py").read_text(encoding="utf-8")
    respond = scripted(
        [call("write_file", {"path": "stats.py", "content": "broken"}, "c1")],
        [message("done")],
    )
    run_agent(TASK, FILES, respond)
    assert (FILES / "stats.py").read_text(encoding="utf-8") == before


# --- trace file ----------------------------------------------------------------

def test_trace_save_and_load_roundtrip(tmp_path):
    trace = Trace()
    trace.add("task", text="héllo")
    trace.add("tool_result", name="run_tests", ok=True, text="exit_code: 0")
    path = tmp_path / "runs" / "t.jsonl"
    trace.save(path)
    assert Trace.load(path) == trace.records
