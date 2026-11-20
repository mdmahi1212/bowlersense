"""The coach: a local Ollama model, with a template report whenever Ollama isn't usable.

Only the JSON metrics summary is sent (never video or images), and only to the localhost
Ollama URL from config (validated to be local). It checks ``/api/tags`` first, then calls
``/api/chat`` (``stream: false``) with the configured timeout and retries. Any failure falls
back to the template report, which stands on its own.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from bowlsense.coach.drills import drills_for
from bowlsense.coach.prompts import CoachSummary, chat_context, report_request, system_prompt
from bowlsense.config import CoachSettings
from bowlsense.errors import BowlSenseError

log = logging.getLogger(__name__)
Source = Literal["ollama", "template"]


class OllamaUnavailableError(BowlSenseError):
    """Ollama isn't running, the model isn't pulled, or the call failed."""


class CoachReport(BaseModel):
    """The written report and where it came from."""

    model_config = ConfigDict(frozen=True)

    text: str
    source: Source
    model: str | None = None
    note: str = ""


class ChatTurn(BaseModel):
    """One message in the coach chat."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: str


class OllamaClient:
    """Minimal client for the local Ollama HTTP API."""

    def __init__(
        self, settings: CoachSettings, transport: httpx.BaseTransport | None = None
    ) -> None:
        """Bind to the configured localhost URL."""
        self.settings = settings
        self._client = httpx.Client(
            base_url=settings.ollama_url, timeout=settings.timeout_s, transport=transport
        )

    def close(self) -> None:
        """Release the connection pool."""
        self._client.close()

    def models(self) -> list[str]:
        """Names of the models pulled into Ollama.

        Raises:
            OllamaUnavailableError: if Ollama isn't reachable.
        """
        try:
            r = self._client.get("/api/tags")
            r.raise_for_status()
            return [str(m["name"]) for m in r.json().get("models", [])]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise OllamaUnavailableError(
                f"Ollama is not reachable at {self.settings.ollama_url}"
            ) from exc

    def has_model(self, name: str) -> bool:
        """True if ``name`` (with or without a ``:tag``) is pulled."""
        pulled = self.models()
        base = name.split(":")[0]
        return name in pulled or (":" not in name and f"{base}:latest" in pulled)

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        """One non-streaming chat completion, retried ``coach.retries`` times.

        Raises:
            OllamaUnavailableError: if every attempt fails.
        """
        payload = {"model": self.settings.model, "messages": list(messages), "stream": False}
        last: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            try:
                r = self._client.post("/api/chat", json=payload)
                r.raise_for_status()
                content = str(r.json()["message"]["content"]).strip()
                if not content:
                    raise ValueError("empty reply")
                return content
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                last = exc
                log.warning("ollama chat attempt %d failed: %s", attempt + 1, exc)
        raise OllamaUnavailableError(f"Ollama chat failed: {last}") from last


def limit_words(text: str, max_words: int) -> str:
    """Trim to at most ``max_words`` words, ending at a sentence boundary when possible."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    cut = " ".join(words[:max_words])
    end = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("! "), cut.rfind("? "))
    return (cut[: end + 1] if end > len(cut) // 2 else cut.rstrip(",;:") + "...").strip()


def _num(value: float) -> str:
    return f"{value:.0f}" if abs(value) >= 10 else f"{value:.1f}"


def template_report(summary: CoachSummary) -> str:
    """A complete report without an LLM: intro, top issues with numbers and drills, risk,
    disclaimers."""
    s = summary
    lines = [
        f"You bowled {s.deliveries} deliver{'y' if s.deliveries == 1 else 'ies'} "
        f"({s.bowling_arm}-arm, {s.view} view). Average technique score: "
        f"{s.technique_score_mean:.0f}/100.",
    ]
    if s.closest_reference:
        lines.append(f"Your action is closest to the {s.closest_reference.lower()}.")
    if s.top_issues:
        lines.append("")
        lines.append(f"Your top {len(s.top_issues)} thing(s) to work on:")
        for i, issue in enumerate(s.top_issues, 1):
            seen = f"in {issue.deliveries_affected} of {s.deliveries} deliveries"
            lines.append(
                f"{i}. {issue.name} ({seen}): averaging {_num(issue.mean_value)} {issue.unit} "
                f"against a limit of {_num(issue.threshold)}. {issue.explanation}"
            )
            for drill in drills_for(issue.code)[:2]:
                lines.append(f"   - Drill: {drill.name} ({drill.reps}). {drill.description}")
    else:
        lines += ["", "No technique faults were found in the checks this camera view allows. "
                      "Keep doing what you're doing."]  # fmt: skip
    lines.append("")
    if s.legality_flagged:
        lines.append(
            f"Elbow: {s.legality_flagged} deliver{'y' if s.legality_flagged == 1 else 'ies'} "
            f"showed more than {s.legality_threshold_deg:.0f} deg of elbow straightening "
            f"(most {s.legality_max_extension_deg:.1f} deg). {s.legality_note}"
        )
    else:
        lines.append(
            f"Elbow: every delivery stayed within {s.legality_threshold_deg:.0f} deg "
            f"(most {s.legality_max_extension_deg:.1f} deg). {s.legality_note}"
        )
    drivers = "; ".join(s.injury_top_drivers[:3]) or "none elevated"
    lines.append(
        f"Injury-risk screening: average {s.injury_overall_mean:.0f}/100, worst band "
        f"{s.injury_band_worst}. Main drivers: {drivers}."
    )
    if s.ball_speeds_kmh:
        lines.append(
            f"Ball speed: {min(s.ball_speeds_kmh):.0f}-{max(s.ball_speeds_kmh):.0f} km/h "
            f"(average {sum(s.ball_speeds_kmh) / len(s.ball_speeds_kmh):.0f})."
        )
    if s.fatigue_alerts:
        lines.append("Fatigue: " + " ".join(s.fatigue_alerts) + ". Consider a rest.")
    if s.consistency_score is not None:
        lines.append(f"Consistency: {s.consistency_score:.0f}/100 across the session.")
    lines += ["", s.injury_disclaimer + " See a qualified professional about any pain."]
    return "\n".join(lines)


def generate_report(
    summary: CoachSummary, settings: CoachSettings, *, use_llm: bool = True,
    transport: httpx.BaseTransport | None = None,
) -> CoachReport:  # fmt: skip
    """The coaching report from the local LLM, or the template when it isn't available."""
    template = template_report(summary)
    if not use_llm:
        return CoachReport(text=template, source="template", note="LLM turned off (--no-llm)")
    client = OllamaClient(settings, transport)
    try:
        if not client.has_model(settings.model):
            return CoachReport(text=template, source="template",
                               note=f"model '{settings.model}' is not pulled in Ollama "
                                    f"(run `ollama pull {settings.model}`)")  # fmt: skip
        text = client.chat(
            [
                {
                    "role": "system",
                    "content": system_prompt(settings.max_words, settings.top_issues),
                },
                {"role": "user", "content": report_request(summary)},
            ]
        )
        return CoachReport(
            text=limit_words(text, settings.max_words), source="ollama", model=settings.model
        )
    except OllamaUnavailableError as exc:
        return CoachReport(text=template, source="template", note=str(exc))
    finally:
        client.close()


def coach_reply(
    summary: CoachSummary,
    history: Sequence[ChatTurn],
    message: str,
    settings: CoachSettings,
    transport: httpx.BaseTransport | None = None,
) -> CoachReport:
    """Answer a chat message with the session's metrics as context.

    Without Ollama, the reply explains that and points to the most relevant part of the
    template report (the issue the message mentions, or the top issue).
    """
    if not message.strip():
        raise ValueError("The chat message is empty.")
    client = OllamaClient(settings, transport)
    turns = list(history)[-2 * settings.chat_history_turns :]
    try:
        if not client.has_model(settings.model):
            raise OllamaUnavailableError(f"model '{settings.model}' is not pulled")
        messages = [{"role": "system", "content": chat_context(summary)}]
        messages += [{"role": t.role, "content": t.content} for t in turns]
        messages.append({"role": "user", "content": message})
        return CoachReport(text=client.chat(messages), source="ollama", model=settings.model)
    except OllamaUnavailableError as exc:
        return CoachReport(text=_offline_answer(summary, message), source="template", note=str(exc))
    finally:
        client.close()


def _offline_answer(summary: CoachSummary, message: str) -> str:
    words = set(re.findall(r"[a-z]+", message.lower()))
    match = next(
        (i for i in summary.top_issues if words & set(re.findall(r"[a-z]+", i.name.lower()))),
        summary.top_issues[0] if summary.top_issues else None,
    )
    head = ("The chat coach needs the local Ollama model, which isn't running right now, so "
            "here is what your numbers say. ")  # fmt: skip
    if match is None:
        return head + "No technique faults were found in this session."
    drill = drills_for(match.code)[:1]
    tip = f" Try: {drill[0].name} ({drill[0].reps}). {drill[0].description}" if drill else ""
    return (
        f"{head}{match.name}: averaging {_num(match.mean_value)} {match.unit} against a limit "
        f"of {_num(match.threshold)}. {match.explanation}{tip}"
    )
