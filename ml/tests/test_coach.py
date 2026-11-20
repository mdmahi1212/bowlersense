"""Tests for the coach (Phase 11): drills, summary, Ollama client (mocked) and template fallback."""

from __future__ import annotations

import json
import logging
from functools import cache

import httpx
import pytest
import respx

from bowlsense.analysis.compare import consistency
from bowlsense.analysis.fatigue import assess_fatigue
from bowlsense.coach.drills import DRILLS, drills_for
from bowlsense.coach.llm import (
    ChatTurn,
    OllamaClient,
    OllamaUnavailableError,
    coach_reply,
    generate_report,
    limit_words,
    template_report,
)
from bowlsense.coach.prompts import CoachSummary, summarise_session, system_prompt
from bowlsense.config import CoachSettings, Settings, is_local_url, load_settings
from bowlsense.pipeline import analyze_delivery
from synth.skeleton_gen import SynthParams, generate

logging.getLogger("bowlsense.phases.heuristic").setLevel(logging.ERROR)

FAULT_CODES = {
    "KNEE_COLLAPSE", "OVERSTRIDE", "UNDERSTRIDE", "HEAD_FALL", "LATERAL_FLEXION",
    "WEAK_FRONT_ARM", "RUNUP_DECEL", "LOW_RELEASE", "CHAIN_ORDER", "SHORT_FOLLOW",
}  # fmt: skip


@cache
def _settings() -> Settings:
    return load_settings()


def _coach(**kw: object) -> CoachSettings:
    return _settings().coach.model_copy(update=kw)


@cache
def _summary(faulty: bool = True) -> CoachSummary:
    s = _settings()
    params = (
        [dict(knee_flex_change_deg=32.0, stride_pct=96.0),
         dict(knee_flex_change_deg=28.0, elbow_extension_deg=22.0), dict(stride_pct=97.0)]
        if faulty else [{}, {}]
    )  # fmt: skip
    ds = [analyze_delivery(generate(SynthParams.model_validate(p)).sequence, s) for p in params]
    fatigue = assess_fatigue([d.fatigue_input for d in ds], s.fatigue)
    return summarise_session(ds, fatigue, consistency(ds, s.compare), "side", "right", s)


def _tags(*names: str) -> httpx.Response:
    return httpx.Response(200, json={"models": [{"name": n} for n in names]})


def _reply(text: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})


# --- drills and summary -------------------------------------------------------------------------


def test_every_fault_has_2_to_3_drills() -> None:
    assert set(DRILLS) == FAULT_CODES
    assert all(2 <= len(drills_for(c)) <= 3 for c in FAULT_CODES)
    assert drills_for("NOPE") == ()


def test_summary_ranks_issues_and_counts() -> None:
    s = _summary()
    assert s.deliveries == 3 and len(s.top_issues) <= _settings().coach.top_issues
    codes = [i.code for i in s.top_issues]
    assert codes[0] == "KNEE_COLLAPSE"  # weight 1.5, in 2 deliveries, high severity
    assert "OVERSTRIDE" in codes
    knee = s.top_issues[0]
    assert knee.deliveries_affected == 2 and knee.mean_value > knee.threshold == 20.0
    assert s.legality_flagged == 1 and s.legality_max_extension_deg > 15.0
    assert s.injury_band_worst in {"low", "medium", "high"}


def test_summary_needs_deliveries() -> None:
    s = _settings()
    with pytest.raises(ValueError, match="No deliveries"):
        summarise_session([], assess_fatigue([], s.fatigue), consistency([], s.compare),
                          "side", "right", s)  # fmt: skip


# --- template -----------------------------------------------------------------------------------


def test_template_report_is_complete() -> None:
    text = template_report(_summary())
    assert "3 deliveries" in text and "Front knee collapse" in text
    assert "Drill:" in text and "against a limit of 20" in text
    assert "2D video screening only" in text and "not medical advice" in text
    assert "Injury-risk screening" in text


def test_template_report_without_faults() -> None:
    text = template_report(_summary(faulty=False))
    assert "No technique faults" in text and "not medical advice" in text


def test_limit_words() -> None:
    long = "One two three. Four five six seven. Eight nine ten eleven twelve."
    assert limit_words(long, 100) == long
    assert limit_words(long, 8) == "One two three. Four five six seven."
    assert limit_words("a b c d e f", 3) == "a b c..."
    listed = "Work on these. 1. Brace the knee firmly. 2. Shorten the stride. 3. Stay tall"
    assert (
        limit_words(listed, 13) == "Work on these. 1. Brace the knee firmly. 2. Shorten the stride."
    )


def test_system_prompt_rules() -> None:
    p = system_prompt(250, 3)
    assert "250 words" in p and "top 3" in p and "medical diagnosis" in p and "numbers" in p


# --- Ollama mocked ------------------------------------------------------------------------------


def test_ollama_url_is_local() -> None:
    assert is_local_url(_settings().coach.ollama_url)


@respx.mock
def test_report_from_ollama_when_running() -> None:
    cfg = _coach()
    base = cfg.ollama_url.rstrip("/")
    respx.get(f"{base}/api/tags").mock(return_value=_tags(cfg.model))
    chat = respx.post(f"{base}/api/chat").mock(return_value=_reply("Brace that front knee."))
    rep = generate_report(_summary(), cfg)
    assert rep.source == "ollama" and rep.model == cfg.model
    assert rep.text == "Brace that front knee."
    body = json.loads(chat.calls.last.request.content)
    assert body["stream"] is False and body["model"] == cfg.model
    assert body["messages"][0]["role"] == "system"
    assert "KNEE_COLLAPSE" in body["messages"][1]["content"]  # only the JSON metrics are sent


@respx.mock
def test_long_llm_reply_is_trimmed() -> None:
    cfg = _coach(max_words=20)
    base = cfg.ollama_url.rstrip("/")
    respx.get(f"{base}/api/tags").mock(return_value=_tags(cfg.model))
    respx.post(f"{base}/api/chat").mock(return_value=_reply("Keep going strong now. " * 30))
    assert len(generate_report(_summary(), cfg).text.split()) <= 20


@respx.mock
def test_fallback_when_ollama_stopped() -> None:
    cfg = _coach()
    respx.get(f"{cfg.ollama_url.rstrip('/')}/api/tags").mock(
        side_effect=httpx.ConnectError("refused")
    )
    rep = generate_report(_summary(), cfg)
    assert rep.source == "template" and "not reachable" in rep.note
    assert rep.text == template_report(_summary())


@respx.mock
def test_fallback_when_model_not_pulled() -> None:
    cfg = _coach()
    respx.get(f"{cfg.ollama_url.rstrip('/')}/api/tags").mock(return_value=_tags("other:1b"))
    rep = generate_report(_summary(), cfg)
    assert rep.source == "template" and "ollama pull" in rep.note


@respx.mock
def test_chat_retries_then_falls_back() -> None:
    cfg = _coach(retries=1)
    base = cfg.ollama_url.rstrip("/")
    respx.get(f"{base}/api/tags").mock(return_value=_tags(cfg.model))
    chat = respx.post(f"{base}/api/chat").mock(side_effect=httpx.ReadTimeout("slow"))
    rep = generate_report(_summary(), cfg)
    assert chat.call_count == 2 and rep.source == "template"


@respx.mock
def test_retry_recovers() -> None:
    cfg = _coach(retries=1)
    base = cfg.ollama_url.rstrip("/")
    respx.get(f"{base}/api/tags").mock(return_value=_tags(cfg.model))
    respx.post(f"{base}/api/chat").mock(
        side_effect=[httpx.Response(500), _reply("Second time lucky.")]
    )
    assert generate_report(_summary(), cfg).text == "Second time lucky."


def test_no_llm_never_calls_ollama() -> None:
    with respx.mock(assert_all_called=False) as router:
        route = router.route().mock(side_effect=AssertionError("network used"))
        rep = generate_report(_summary(), _coach(), use_llm=False)
    assert rep.source == "template" and not route.called


@respx.mock
def test_tag_matching() -> None:
    cfg = _coach(model="llama3.1")
    respx.get(f"{cfg.ollama_url.rstrip('/')}/api/tags").mock(return_value=_tags("llama3.1:latest"))
    client = OllamaClient(cfg)
    try:
        assert client.has_model("llama3.1") and not client.has_model("llama3.1:8b")
    finally:
        client.close()


@respx.mock
def test_models_raises_on_bad_payload() -> None:
    cfg = _coach()
    respx.get(f"{cfg.ollama_url.rstrip('/')}/api/tags").mock(
        return_value=httpx.Response(200, text="not json")
    )
    client = OllamaClient(cfg)
    with pytest.raises(OllamaUnavailableError):
        client.models()
    client.close()


# --- chat ---------------------------------------------------------------------------------------


@respx.mock
def test_chat_keeps_session_context_and_history() -> None:
    cfg = _coach(chat_history_turns=1)
    base = cfg.ollama_url.rstrip("/")
    respx.get(f"{base}/api/tags").mock(return_value=_tags(cfg.model))
    chat = respx.post(f"{base}/api/chat").mock(return_value=_reply("Land on a firm leg."))
    history = [
        ChatTurn(role="user", content="old q"), ChatTurn(role="assistant", content="old a"),
        ChatTurn(role="user", content="q1"), ChatTurn(role="assistant", content="a1"),
    ]  # fmt: skip
    rep = coach_reply(_summary(), history, "How do I fix my knee?", cfg)
    assert rep.source == "ollama" and rep.text == "Land on a firm leg."
    msgs = json.loads(chat.calls.last.request.content)["messages"]
    assert msgs[0]["role"] == "system" and "KNEE_COLLAPSE" in msgs[0]["content"]
    assert [m["content"] for m in msgs[1:]] == ["q1", "a1", "How do I fix my knee?"]


@respx.mock
def test_chat_offline_answer_matches_question() -> None:
    cfg = _coach()
    respx.get(f"{cfg.ollama_url.rstrip('/')}/api/tags").mock(
        side_effect=httpx.ConnectError("refused")
    )
    rep = coach_reply(_summary(), [], "Is my stride too long?", cfg)
    assert rep.source == "template" and "Over-long delivery stride" in rep.text
    assert "Drill" not in rep.text and "Try:" in rep.text
    with pytest.raises(ValueError, match="empty"):
        coach_reply(_summary(), [], "  ", cfg)
