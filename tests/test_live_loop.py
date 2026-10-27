"""Tests the live agent loop with a scripted fake Claude client (no API key, no network).

Run with:  pytest -q
"""
from types import SimpleNamespace as NS

import anthropic

from backend import agent


def text(t):
    return NS(type="text", text=t)


def tool(name, inp, id_="t1"):
    return NS(type="tool_use", name=name, input=inp, id=id_)


def resp(*blocks, tin=10, tout=5):
    return NS(content=list(blocks), usage=NS(input_tokens=tin, output_tokens=tout))


class FakeClient:
    def __init__(self, script):
        self.script, self.calls, self.messages = list(script), [], self

    def create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)


def fake_search(query, top_k=5, date_range=None):
    return {"results": [{"title": "Real A", "url": "https://real.org/a", "snippet": "x"}],
            "retrieved_at": "now"}


def fake_read(url, focus_question=""):
    return {"title": "Real A", "url": url, "passages": [{"text": "p", "location": "chunk 1"}],
            "retrieved_at": "now"}


REPORT = {"findings": ["first finding", "second finding"]}


def install(monkeypatch, script):
    client = FakeClient(script)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: client)
    monkeypatch.setitem(agent.HANDLERS, "web_search", fake_search)
    monkeypatch.setitem(agent.HANDLERS, "read_document", fake_read)
    return client


def test_full_loop_search_read_report(monkeypatch):
    install(monkeypatch, [
        resp(text("Plan: search then read."), tool("web_search", {"query": "q"}, "a")),
        resp(tool("read_document", {"url": "https://real.org/a", "focus_question": "q"}, "b")),
        resp(tool("submit_report", REPORT, "c")),
    ])
    events = list(agent.run_agent("question", "live", 6))
    kinds = [e["type"] for e in events]
    assert "error" not in kinds and kinds[-1] == "metrics"
    report = next(e["report"] for e in events if e["type"] == "report")
    assert report == {"findings": ["first finding", "second finding"]}
    m = events[-1]
    assert m["steps"] == 2 and m["input_tokens"] == 30 and m["output_tokens"] == 15
    assert events[0]["phase"] == "plan"


def test_budget_exhausted_forces_report(monkeypatch):
    client = install(monkeypatch, [
        resp(tool("web_search", {"query": "q"}, "a")),
        resp(tool("submit_report", REPORT, "z")),  # answer to the forced call
    ])
    events = list(agent.run_agent("question", "live", 1))
    assert any(e["type"] == "report" for e in events)
    assert any(e["type"] == "step" and e["title"] == "Step budget reached" for e in events)
    forced = client.calls[-1]
    assert forced["tool_choice"] == {"type": "tool", "name": "submit_report"}
    last_user = forced["messages"][-1]
    assert last_user["role"] == "user" and any(b.get("type") == "text" for b in last_user["content"])


def test_unknown_tool_is_blocked(monkeypatch):
    install(monkeypatch, [
        resp(tool("delete_everything", {}, "a")),
        resp(tool("submit_report", REPORT, "b")),
    ])
    events = list(agent.run_agent("question", "live", 6))
    assert any("Blocked unknown tool" in e.get("title", "") for e in events)
    assert any(e["type"] == "report" for e in events)


def test_api_failure_becomes_error_event(monkeypatch):
    class Boom:
        messages = None

        def __init__(self):
            self.messages = self

        def create(self, **kw):
            raise RuntimeError("rate limited")

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: Boom())
    events = list(agent.run_agent("question", "live", 3))
    assert events[-1]["type"] == "error" and "rate limited" in events[-1]["message"]
