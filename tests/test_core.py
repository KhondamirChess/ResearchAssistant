"""Run with:  pytest -q   (from the project root, venv active). No API keys needed."""
from backend import agent, tools


def test_blocks_local_and_bad_urls():
    assert not tools.is_safe_url("http://localhost:8000/secret")
    assert not tools.is_safe_url("http://127.0.0.1/")
    assert not tools.is_safe_url("http://169.254.169.254/latest/meta-data")
    assert not tools.is_safe_url("file:///etc/passwd")
    assert not tools.is_safe_url("not a url")


def test_select_passages_prefers_relevant_text():
    text = ("Cats sleep a lot during the day and enjoy warm places to rest.\n\n"
            "Retrieval augmented generation combines a retriever with a language model "
            "so answers can cite documents.\n\n"
            "The weather in spring is usually mild with occasional rain showers.")
    out = tools.select_passages(text, "retrieval augmented generation", n=1)
    assert len(out) == 1 and "Retrieval augmented" in out[0]["text"]


def test_empty_search_query_returns_error_not_crash():
    assert "error" in tools.web_search("   ")


def test_reader_rejects_private_url():
    assert "error" in tools.read_document("http://localhost/x", "anything")


def test_normalize_report_keeps_clean_findings_only():
    raw = {"findings": ["  first  ", "", {"text": "second"}, None, 42]}
    assert agent.normalize_report(raw) == {"findings": ["first", "second", "42"]}
    assert agent.normalize_report({}) == {"findings": []}
    assert agent.normalize_report("junk") == {"findings": []}


def test_mock_data_has_every_language_for_every_item():
    data = agent.load_mock()
    assert len(data["items"]) >= 6
    for item in data["items"]:
        for lang in agent.LANGS:
            assert item["questions"][lang].strip()
            assert len(item["findings"][lang]) == len(item["findings"]["en"]) >= 4
    for lang in agent.LANGS:
        assert "{topic}" in data["fallback"][lang][0]


def test_every_mock_question_matches_its_own_item_in_every_language():
    for item in agent.load_mock()["items"]:
        for q in item["questions"].values():
            assert agent.match_mock(q)["id"] == item["id"]


def test_match_mock_tolerates_small_typos_and_rejects_unrelated_text():
    assert agent.match_mock("what are main limitations of retrieval augmented generation")["id"] == "rag-limitations"
    assert agent.match_mock("best pizza recipe in naples") is None


def test_demo_run_produces_report_and_metrics(monkeypatch):
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    events = list(agent.run_agent("What is RAG?", "demo", 6))
    kinds = [e["type"] for e in events]
    assert "report" in kinds and kinds[-1] == "metrics" and "error" not in kinds
    phases = {e["phase"] for e in events if e["type"] == "step"}
    assert {"plan", "act", "observe", "reflect", "write"} <= phases


def test_demo_returns_findings_list_in_selected_language(monkeypatch):
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    item = agent.load_mock()["items"][1]
    for lang in agent.LANGS:
        events = list(agent.run_agent(item["questions"][lang], "demo", 6, lang))
        report = next(e["report"] for e in events if e["type"] == "report")
        assert list(report) == ["findings"]
        assert report["findings"] == item["findings"][lang]


def test_demo_unknown_question_uses_fallback_with_topic(monkeypatch):
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    events = list(agent.run_agent("best pizza recipe in naples?", "demo", 6, "ko"))
    report = next(e["report"] for e in events if e["type"] == "report")
    assert "best pizza recipe in naples" in report["findings"][0]
