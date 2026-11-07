"""Basic tests. Run with:  pytest -q"""
from src import llm, rag
from src.ingest import chunk_text, clean_text, load_jobs, make_chunks
from src.retriever import Retriever
from src.skills import extract_skills


def test_chunking_overlap():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]
    assert chunks[-1].split()[-1] == "99"


def test_clean_text_removes_boilerplate():
    text = "Requirements\n- Python\nEqual opportunities statement here"
    assert "Equal" not in clean_text(text)


def test_chunks_keep_metadata():
    records = make_chunks(load_jobs(), 120, 20)
    assert all(r["metadata"]["company"] and r["text"].count(":") >= 1 for r in records)


def test_skill_extraction():
    found = extract_skills("Python, R or SQL; Power BI dashboards; C# and C++; regression")
    assert {"Python", "R", "SQL", "Power BI", "C# / C++", "Regression", "Data visualisation"} <= found
    assert "R" not in extract_skills("We are a great team")  # no false match on the letter r


def test_retrieval_finds_expected_job():
    hits = Retriever("tfidf").search("Which role uses C# and WPF?", k=3)
    assert hits[0]["job_id"] == "renishaw_software_placement"


def test_answer_uses_llm_when_configured(monkeypatch):
    monkeypatch.setattr(llm, "is_configured", lambda: True)
    monkeypatch.setattr(llm, "chat", lambda system, user, **kw: "stub answer")
    result = rag.answer("Which roles need Power BI?", Retriever("tfidf"))
    assert result["answer"] == "stub answer" and result["sources"]


def test_citations_are_normalised():
    raw = ("Yes \u3010Arm - Data Analytics Intern\u3011 and [Faculty\u202f\u2013\u202fData Science Fellowship]; "
           "EY-Parthenon [Rolls\u2011Royce \u2013 Engineering Internship] [ AWE - Safety ]")
    assert rag.normalise_citations(raw) == (
        "Yes [Arm - Data Analytics Intern] and [Faculty - Data Science Fellowship]; "
        "EY-Parthenon [Rolls-Royce - Engineering Internship] [AWE - Safety]")


def test_hybrid_search_finds_keyword_and_paraphrase():
    r = Retriever("hybrid")
    visa = [h["job_id"] for h in r.search("Which role does not accept Graduate or post-study work visas?", k=8)]
    chip = [h["job_id"] for h in r.search("Where would I help a chip design company check the quality of its products?", k=8)]
    assert "sparta_junior_data_consultant" in visa and "arm_data_analytics_intern" in chip


def test_tfidf_returns_nothing_when_no_word_matches():
    # no query word is in the vocabulary, so a ranking would be random
    assert Retriever("tfidf").search("zzqx wvvy", k=5) == []


def _rate_limit_error(message):
    import httpx
    from openai import RateLimitError
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.test"))
    return RateLimitError(message, response=response, body=None)


def _fake_openai(monkeypatch, fail_models, calls):
    """Replace the OpenAI client: models in fail_models raise a daily rate-limit error."""
    import openai
    from types import SimpleNamespace

    def create(model, **kwargs):
        calls.append(model)
        if model in fail_models:
            raise _rate_limit_error("Rate limit reached on tokens per day (TPD). Please try again in 12m8.7s.")
        message = SimpleNamespace(content="<think>hidden</think>Answer [Arm - Data Analytics Intern]")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: client)


def test_llm_falls_back_to_backup_model(monkeypatch):
    from src import config
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "LLM_MODEL", "main-model")
    monkeypatch.setattr(config, "LLM_FALLBACK_MODELS", ["backup-model"])
    calls = []
    _fake_openai(monkeypatch, {"main-model"}, calls)
    assert llm.chat("system", "user") == "Answer [Arm - Data Analytics Intern]"
    assert calls == ["main-model", "backup-model"] and llm.last_model == "backup-model"


def test_answer_shows_passages_when_every_model_is_limited(monkeypatch):
    from src import config
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_MODEL", "main-model")
    monkeypatch.setattr(config, "LLM_FALLBACK_MODELS", ["backup-model"])
    _fake_openai(monkeypatch, {"main-model", "backup-model"}, [])
    result = rag.answer("Which role uses C# and WPF?", Retriever("tfidf"))
    assert "daily usage limit reached, try again in 12m8.7s" in result["notice"]
    assert result["model"] is None and "most relevant passages" in result["answer"]
