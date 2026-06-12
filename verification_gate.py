"""Verification Gate — VLM confirmation layer that fires only when a rule matches.

Receives: current frame + candidate alert + scene context
Returns:  confirmed/rejected with a reason and confidence score

Cost note: only called when the customization engine fires, not every frame.
~$0.001 per alert call is acceptable; ~$0.001 per frame is not.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2

if TYPE_CHECKING:
    from customization import CandidateAlert


@dataclass
class VerificationResult:
    confirmed: bool
    confidence: float
    reason: str
    alert_priority: str
    timestamp: str
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed": self.confirmed,
            "confidence": self.confidence,
            "reason": self.reason,
            "alert_priority": self.alert_priority,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are a security alert verification assistant for a CCTV system.

A computer vision system has flagged a potential security event. You are shown one or more camera frames from the SAME short event (a brief sequence in time) — use the motion across them to decide if the alert is genuine.

Scene context:
- Environment: {environment_type}
- Description: {scene_description}

Candidate alert:
- Rule: {rule_name}
- Detector: {detector}
- Detection: {title}
- Person tracked: {person_id}
- Object involved: {object_label}

Question: {question}

Respond with a single JSON object only. No markdown. No text before or after the JSON.

{{
  "confirmed": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "one sentence explaining your decision based on what you see",
  "alert_priority": "{priority}"
}}

Rules:
- Only confirm if the visual evidence clearly supports the alert.
- If the scene looks normal or the alert is ambiguous, return confirmed: false.
- confidence should reflect how certain you are, not how alarming the scene is.
"""

_QUESTIONS: dict[str, str] = {
    "shoplifting": "Does this frame show a person concealing or taking merchandise without paying in a {environment_type}?",
    "violence_in_store": "Does this frame show genuine physical violence or assault in a {environment_type}?",
    "weapon_sighting": "Does this frame show a real weapon being carried or brandished by a person?",
    "after_hours_intrusion": "Does this frame show unauthorized presence in a {environment_type} outside business hours?",
    "loitering_near_merchandise": "Does this frame show a person behaving suspiciously near merchandise in a {environment_type}?",
    "armed_robbery": "Does this frame show an armed robbery in progress?",
    "assault_at_branch": "Does this frame show an assault at a banking branch?",
    "theft_attempt": "Does this frame show a person attempting to steal something?",
    "card_skimming_suspect": "Does this frame show suspicious behavior at an ATM or card reader?",
    "after_hours_presence": "Does this frame show unauthorized presence during closed hours?",
}


def _build_question(rule_name: str, environment_type: str) -> str:
    template = _QUESTIONS.get(
        rule_name,
        "Does this frame confirm a security threat event in a {environment_type}?",
    )
    return template.format(environment_type=environment_type)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VerificationGate:
    DEFAULT_MODEL = {"anthropic": "claude-sonnet-4-6", "openrouter": "google/gemma-4-26b-a4b-it:free"}
    DEFAULT_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}

    def __init__(
        self,
        provider: str = "mock",
        model: str = "",
        api_key_env: str = "ANTHROPIC_API_KEY",
        save_dir: Path | str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model or self.DEFAULT_MODEL.get(provider, "claude-sonnet-4-6")
        # If the caller left the default Anthropic key env but picked another provider,
        # switch to that provider's conventional env var (e.g. OPENROUTER_API_KEY).
        if api_key_env == "ANTHROPIC_API_KEY" and provider in self.DEFAULT_KEY_ENV:
            api_key_env = self.DEFAULT_KEY_ENV[provider]
        self.api_key_env = api_key_env
        self.save_dir = Path(save_dir) if save_dir else None
        self._call_count = 0

    def verify(
        self,
        frame: Any,               # a single numpy BGR frame OR a list of them (multi-frame)
        alert: CandidateAlert,
        scene_context: dict | None = None,
    ) -> VerificationResult:
        self._call_count += 1
        context = scene_context or {}
        environment_type = context.get("environment_type", "unknown")
        scene_description = context.get("scene_description", "No scene description available.")

        prompt = _PROMPT_TEMPLATE.format(
            environment_type=environment_type,
            scene_description=scene_description,
            rule_name=alert.rule_name,
            detector=alert.detector,
            title=alert.title,
            person_id=alert.person_id if alert.person_id is not None else "unknown",
            object_label=alert.object_label or "unknown",
            question=_build_question(alert.rule_name, environment_type),
            priority=alert.priority,
        )

        frames = frame if isinstance(frame, list) else [frame]
        frames_bytes = [_encode_frame(f) for f in frames]

        if self.provider == "mock":
            raw_response = _mock_response(alert)
        elif self.provider == "anthropic":
            raw_response = _call_anthropic(prompt, frames_bytes, self.model, self.api_key_env)
        elif self.provider == "openrouter":
            raw_response = _call_openrouter(prompt, frames_bytes, self.model, self.api_key_env)
        else:
            raise RuntimeError(f"Unsupported provider: {self.provider}")

        result = _parse_response(raw_response, alert.priority)

        if self.save_dir:
            _save_artifacts(self.save_dir, self._call_count, frames, alert, result, raw_response)

        return result


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def _encode_frame(frame: Any) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return encoded.tobytes()


def _mock_response(alert: CandidateAlert) -> str:
    # Mock always confirms active alerts — useful for testing the wiring without an API key.
    return json.dumps({
        "confirmed": True,
        "confidence": 0.72,
        "reason": f"Mock provider: alert {alert.rule_name} accepted for testing.",
        "alert_priority": alert.priority,
    })


def _call_openrouter(prompt: str, frames_bytes: list[bytes], model: str, api_key_env: str) -> str:
    """Verify via an OpenAI-compatible endpoint (OpenRouter / Gemma 4 etc.), one or many frames.

    Reuses agent_mapper.call_openai_compatible — the same retry/backoff path already
    validated against OpenRouter's free tier — so we don't duplicate the HTTP logic.
    """
    from agent_mapper import call_openai_compatible  # local import: only when used
    return call_openai_compatible(
        prompt=prompt,
        frame_bytes=frames_bytes,
        model=model,
        api_key_env=api_key_env,
        api_base_url="https://openrouter.ai/api/v1",
    )


def _call_anthropic(prompt: str, frames_bytes: list[bytes], model: str, api_key_env: str) -> str:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"ANTHROPIC_API_KEY not set. Export it with: export {api_key_env}=your_key"
        )

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for fb in frames_bytes:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": base64.b64encode(fb).decode("ascii")},
        })
    payload = {
        "model": model,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": content}],
    }
    req = urlrequest.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Anthropic request failed HTTP {exc.code}: {detail}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"Anthropic request failed: {exc}") from exc

    chunks = [item.get("text", "") for item in body.get("content", []) if item.get("type") == "text"]
    if not chunks:
        raise RuntimeError("Anthropic response contained no text.")
    return "\n".join(chunks).strip()


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    if start < 0:
        raise RuntimeError("No JSON object in VLM response.")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise RuntimeError("Unterminated JSON in VLM response.")


def _parse_response(raw: str, fallback_priority: str) -> VerificationResult:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        data = _extract_json(raw)
        return VerificationResult(
            confirmed=bool(data.get("confirmed", False)),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            reason=str(data.get("reason", "")).strip() or "No reason provided.",
            alert_priority=str(data.get("alert_priority", fallback_priority)),
            timestamp=timestamp,
            raw_response=raw,
        )
    except Exception as exc:
        return VerificationResult(
            confirmed=False,
            confidence=0.0,
            reason=f"Gate parse error: {exc}",
            alert_priority=fallback_priority,
            timestamp=timestamp,
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def _save_artifacts(
    save_dir: Path,
    call_count: int,
    frames: list,
    alert: CandidateAlert,
    result: VerificationResult,
    raw_response: str,
) -> None:
    out_dir = save_dir / f"gate_{call_count:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        cv2.imwrite(str(out_dir / (f"frame_{i}.jpg" if len(frames) > 1 else "frame.jpg")), fr)
    (out_dir / "alert.json").write_text(
        json.dumps(alert.to_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "verification.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
