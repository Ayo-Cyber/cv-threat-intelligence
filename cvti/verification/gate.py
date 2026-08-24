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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2

from cvti.contracts import CandidateAlert, VerificationResult

from cvti.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are the FINAL verification check for a CCTV security system. A cheap computer-vision
detector has flagged a possible event. That detector is frequently WRONG — it over-fires
on normal activity. Your job is to catch its mistakes, not to agree with it.

You are shown one or more camera frames from the SAME short event (a brief sequence in
time) — use the motion across them to decide for yourself what is happening.

Scene context:
- Environment: {environment_type}
- Description: {scene_description}

The detector's UNRELIABLE hypothesis (treat as a claim to be disproven, NOT a fact):
- Rule: {rule_name}
- Detector: {detector}
- Claimed detection: {title}
- Person tracked: {person_id}
- Object involved: {object_label}

Question: {question}

Respond with a single JSON object only. No markdown. No text before or after the JSON.

{{
  "confirmed": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "one sentence explaining your decision based ONLY on what you see",
  "alert_priority": "{priority}"
}}

You are TRIAGING for a human reviewer, not making a courtroom judgement.
Rules:
- Do NOT rubber-stamp the detector's label, and do NOT invent innocent excuses for
  everything either. Judge the frames as they are.
- REJECT when the frames show plainly ordinary behaviour with no sign of the threat
  (standing, walking, browsing, using a phone/laptop, working a till, an empty scene).
  This is how you kill the detector's false positives.
- CONFIRM when the frames are genuinely consistent with the SPECIFIC threat in the
  question — you do not need absolute proof, only a real, visible reason to escalate it
  to a human. A missed real threat is worse than a reviewed false alarm.
- If the frames could plausibly be the threat OR innocent and you truly cannot tell,
  lean CONFIRM (escalate for review) — but only when there is an actual visible cue,
  not merely because the detector said so.
- confidence reflects certainty in your OWN verdict, not how alarming the scene is.
"""

_COT_PROMPT_TEMPLATE = """\
You are the verification step for a CCTV security system. A detector has flagged a possible
event. Your job is to PASS REAL THREATS THROUGH to a human reviewer and filter out only the
OBVIOUS false alarms. Missing a real threat is much worse than passing on a scene the human
glances at and dismisses — so when in doubt, confirm.
You are shown one or more frames from the SAME short event (a brief sequence in time).

Scene context:
- Environment: {environment_type}
- Description: {scene_description}

The detector's report (usually right; you are just catching its occasional mistakes):
- Rule: {rule_name}
- Detector: {detector}
- Detection: {title}
- Person tracked: {person_id}
- Object involved: {object_label}

Question: {question}

Reason briefly FIRST (plain text, no JSON yet):
1. Are there any PEOPLE in these frames? If not, the answer is almost always false.
2. What is the person actually doing — describe only what you can literally SEE (hands, body,
   what they are holding or touching). Do not describe what the detector claims.
3. Does what you see match the SPECIFIC threat in the question, or is it ordinary activity
   (standing, walking, queueing, browsing, working a counter, an empty street)?

Then on the FINAL line, output ONLY this JSON object (no markdown, nothing after it):
{{"confirmed": true or false, "confidence": 0.0 to 1.0, "reason": "one sentence naming the
visible evidence, or why there is none", "alert_priority": "{priority}"}}

Confirm ONLY when you can name a concrete visible cue of that specific threat. Ordinary
activity and empty scenes are false — rejecting them is the job. Do not confirm merely
because the detector flagged it, and do not invent detail you cannot see. Set confidence to
how sure you are of YOUR OWN verdict: high when the evidence is unmistakable, low when you
are guessing.
"""

_QUESTIONS: dict[str, str] = {
    "shoplifting": ("Is a person CONCEALING merchandise in this {environment_type} — slipping it "
                    "into clothing, a pocket, a waistband, or a personal bag? Handling, examining "
                    "or carrying goods openly, or putting them in a shop basket or trolley, is "
                    "normal shopping and is NOT shoplifting."),
    "violence_in_store": "Does this frame show genuine physical violence or assault in a {environment_type}?",
    "weapon_sighting": "Does this frame show a real weapon being carried or brandished by a person?",
    "after_hours_intrusion": "Does this frame show unauthorized presence in a {environment_type} outside business hours?",
    "loitering_near_merchandise": "Does this frame show a person behaving suspiciously near merchandise in a {environment_type}?",
    "armed_robbery": "Does this frame show an armed robbery in progress?",
    "assault_at_branch": "Does this frame show an assault at a banking branch?",
    "theft_attempt": "Does this frame show a person attempting to steal something?",
    "card_skimming_suspect": "Does this frame show suspicious behavior at an ATM or card reader?",
    "after_hours_presence": "Does this frame show unauthorized presence during closed hours?",
    "camera_tampering": "Is this camera view blacked out, physically covered, obstructed, sprayed, or defocused — i.e. deliberately tampered with or blocked? Sudden darkness from lights turning off is NOT tampering.",
    "camera_blocked": "Is this camera view blacked out, physically covered, obstructed, or defocused — i.e. deliberately tampered with or blocked?",
    # HSE situational rules (see cvti.detector.situational)
    "baseline_fire_smoke": "Does this frame show visible fire, flames, smoke, or hazardous haze in a {environment_type}?",
    "fire_smoke": "Does this frame show visible fire, flames, smoke, or hazardous haze in a {environment_type}?",
    "panic_running": "Does this brief sequence show a person running or moving with panic/urgency in a {environment_type}? Someone walking calmly is NOT panic.",
    "unsafe_crowd_formation": "Does this frame show an unsafe crowd or tight group formation in a {environment_type}? A few people spread out normally is NOT unsafe.",
}

# Some detectors carry their meaning in the detector name rather than the rule name;
# fall back to a detector-specific question so the VLM verifies the RIGHT thing.
_DETECTOR_QUESTIONS: dict[str, str] = {
    "weapons": "Does this frame clearly show a real weapon (gun, knife, blade) being held, carried, or brandished by a person? A phone, tool, bottle, or empty hand is NOT a weapon.",
    "camera_tampering": _QUESTIONS["camera_tampering"],
    # The theft question is sensitivity-dependent — see SENSITIVITY_QUESTIONS below.
    # This entry is the 'balanced' default; VerificationGate swaps it per setting.
    "video_action": (
        "An action model flagged this sequence as theft. It over-fires — it flags empty "
        "streets, parked cars and ordinary passers-by — so judge the FRAMES YOURSELF.\n"
        "Answer TRUE only if you can point to a person doing something concrete: taking, "
        "grabbing or carrying off goods; concealing an item on their body or in a bag; "
        "forcing, prying or breaking something open; snatching from a person or vehicle; or "
        "fleeing with property.\n"
        "Answer FALSE if the frames show: no people at all; only vehicles, buildings or an "
        "empty street; people merely present, standing, walking, queueing or browsing; or a "
        "normal transaction at a counter."),
    "fire": "Does this frame show visible fire, flames, smoke, or hazardous haze? Sunset/sunlight, orange or red signage, reflections, screens, and coloured lighting are NOT fire.",
    "person_fall": "Do these frames show a person who has collapsed or is lying on the ground (fallen, fainted, or knocked down) and NOT getting up — a possible medical emergency? Someone sitting, crouching, kneeling, bending down, or deliberately lying down is NOT a fall.",
    "running": "Does this brief sequence show a person running or moving with panic/urgency? Someone walking calmly is NOT panic.",
    "crowd_formation": "Does this frame show an unsafe crowd or tight group formation blocking movement or exits? A few people spread out normally is NOT unsafe.",
}


# ---------------------------------------------------------------------------
# Sensitivity — a measured trade-off, not a guess.
#
# Numbers below are from tests on the 36-clip held-out CamNuvem test split with
# gemma3:4b (runs/eval/*). Tightening the theft question buys precision and costs
# recall, monotonically — so this is the operator's call, not ours.
#
#   sensitive  catch everything, tolerate noise   recall 88.9%  FPR 25.9%  prec 53.3%
#   balanced   (default) same as 'sensitive' today — kept distinct so the
#              wording can diverge without breaking configs
#   strict     quiet console, accept misses       recall 77.8%  FPR 14.8%  prec 63.6%
# ---------------------------------------------------------------------------

_STRICT_THEFT_Q = (
    "An action model flagged this sequence as theft. It over-fires — on empty streets, "
    "parked cars and ordinary shoppers — so judge the FRAMES YOURSELF.\n"
    "CRITICAL: picking up, holding, examining or carrying merchandise is NORMAL SHOPPING, "
    "not theft. Customers take things off shelves; that is the entire point of a shop. "
    "Do NOT confirm because someone is 'taking an item from the shelf'.\n"
    "Answer TRUE only for a visible sign the goods are being STOLEN rather than shopped: "
    "an item being CONCEALED (slipped into clothing, a pocket, a waistband, or a personal "
    "bag — not a shop basket or trolley); something being forced, pried or broken open; "
    "property snatched from a person or vehicle; or someone running away with goods.\n"
    "Answer FALSE for: no people at all; only vehicles, buildings or an empty street; "
    "people standing, walking, queueing, browsing, or handling goods openly; anyone paying "
    "or being served at a counter; or simply walking out carrying a bag.")

_STRICT_SHOPLIFTING_Q = (
    "Is a person CONCEALING merchandise in this {environment_type} — slipping it into "
    "clothing, a pocket, a waistband, or a personal bag? Handling, examining or carrying "
    "goods openly, or putting them in a shop basket or trolley, is normal shopping and is "
    "NOT shoplifting.")

SENSITIVITY_QUESTIONS: dict[str, dict[str, str]] = {
    "sensitive": {},                       # use the defaults above
    "balanced": {},
    "strict": {"video_action": _STRICT_THEFT_Q, "shoplifting": _STRICT_SHOPLIFTING_Q},
}

# Measured on the held-out set — surfaced so the UI/docs can state real numbers.
# --- BEGIN GENERATED: SENSITIVITY_MEASURED (tools/make_sensitivity.py) ---
# Computed from the archived eval runs (runs/eval/v2-tightened, v3-strict),
# not hand-maintained: regenerate with `python tools/make_sensitivity.py`
# after any re-measurement. A test asserts these equal what the archives
# produce, so the code cannot drift from the reality it claims to describe.
SENSITIVITY_MEASURED = {
    "balanced": {"recall": 0.889, "precision": 0.533, "fpr": 0.259, "alerts": 61},
    "strict": {"recall": 0.778, "precision": 0.636, "fpr": 0.148, "alerts": 28},
    "_dataset": "36 held-out CamNuvem test clips (9 threat / 27 normal), gemma3:4b",
}
# --- END GENERATED: SENSITIVITY_MEASURED ---


def _format_examples(examples: list | None) -> str:
    """Turn recent operator feedback into a short few-shot block appended to the
    prompt, so the gate calibrates to THIS site (e.g. learns that a given camera's
    'crowd' flags are usually false)."""
    if not examples:
        return ""
    lines = []
    for e in examples[:4]:
        # accept dicts or LabeledEvent-like objects
        review = getattr(e, "review", None) or (e.get("review") if isinstance(e, dict) else None)
        reason = getattr(e, "reason", None) or (e.get("reason") if isinstance(e, dict) else "")
        if review not in ("true", "false"):
            continue
        verdict = "was a REAL threat" if review == "true" else "was a FALSE alarm"
        lines.append(f'- A prior similar alert here {verdict}: "{str(reason)[:120]}"')
    if not lines:
        return ""
    return ("\n\nRecent operator feedback on THIS camera for this kind of alert "
            "(use it to calibrate — the operator is the ground truth):\n" + "\n".join(lines))


def _build_question(rule_name: str, environment_type: str, detector: str = "",
                    sensitivity: str = "balanced") -> str:
    # A sensitivity preset can override the question for specific rules/detectors.
    override = SENSITIVITY_QUESTIONS.get(sensitivity, {})
    tpl = override.get(rule_name) or override.get(detector)
    if tpl:
        return tpl.format(environment_type=environment_type)
    # Prefer a rule-specific question; else fall back to a detector-specific one so
    # weapons/tamper/video-action get verified for the RIGHT thing (not a generic
    # "is this a threat"). Only then the generic catch-all.
    template = _QUESTIONS.get(rule_name) or _DETECTOR_QUESTIONS.get(detector) or (
        "Does this frame confirm a genuine security threat event in a {environment_type}?"
    )
    return template.format(environment_type=environment_type)


# ---------------------------------------------------------------------------
# Mock-gate safety
# ---------------------------------------------------------------------------
# `_mock_response()` unconditionally returns confirmed=True. That is fine for
# wiring tests and the eval harness, which construct a VerificationGate on
# purpose — and catastrophic for a running engine, where it silently inverts the
# product: every candidate the cheap detectors emit becomes a confirmed alert,
# and nothing on screen says so.
#
# So: constructing a mock gate stays free, but *starting an engine* on one is
# refused unless the operator asked for it by name.

ALLOW_MOCK_GATE_ENV = "ARGUS_ALLOW_MOCK_GATE"
MOCK_GATE_BANNER = "UNVERIFIED — MOCK GATE"


class MockGateRefused(RuntimeError):
    """Raised when an engine is asked to start with the always-confirms gate."""


class UnsupportedProvider(RuntimeError):
    """A provider name nothing knows how to call. A config bug, never transient."""


# What an operator sees when the gate could not reach a verdict. Deliberately
# not phrased as a detection: it says the system does not know.
UNVERIFIED_REASON = "UNVERIFIED — TrueSight could not decide. Review this manually."

# Fail-visible: when verification is impossible, put the alert in front of a
# human rather than discarding it. The alternative — fail-silent — means a real
# fire disappears because Ollama was restarting, and nothing anywhere says so.
# For a safety system the safe default is to interrupt someone.
FAIL_VISIBLE_DEFAULT = True


def mock_gate_allowed() -> bool:
    return os.environ.get(ALLOW_MOCK_GATE_ENV, "").strip() == "1"


def assert_engine_gate_allowed(provider: str) -> bool:
    """Gate-keep engine startup. Returns True iff running on an explicitly
    allowed mock gate (the caller should then show the red banner).

    Raises MockGateRefused for an unannounced mock gate.
    """
    if provider != "mock":
        return False
    if not mock_gate_allowed():
        raise MockGateRefused(
            "Refusing to start: gate provider is 'mock', which confirms EVERY alert "
            "without verifying anything. Use --gate-provider ollama (or anthropic / "
            f"openrouter / local). To run unverified on purpose, set {ALLOW_MOCK_GATE_ENV}=1."
        )
    return True


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VerificationGate:
    # Sensible per-provider default models.
    _DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-6",
        "local": "gemma3:4b-it-qat",
        "openai_compatible": "gpt-4.1-mini",
        "openrouter": "google/gemma-4-26b-a4b-it:free",
        "ollama": "gemma3:4b",
    }
    # Conventional API-key env var per provider.
    DEFAULT_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY",
                       "ollama": "OLLAMA_API_KEY"}

    def __init__(
        self,
        provider: str = "mock",
        model: str = "",
        api_key_env: str = "ANTHROPIC_API_KEY",
        save_dir: Path | str | None = None,
        base_url: str = "",
        cot: bool = True,
        min_confidence: float = 0.35,
        sensitivity: str = "balanced",
        fail_visible: bool | None = None,
    ) -> None:
        self.provider = provider
        self.cot = cot
        # Configurable, because a low-stakes deployment drowning in unverified
        # alerts may reasonably choose otherwise. Default is fail-visible.
        self.fail_visible = FAIL_VISIBLE_DEFAULT if fail_visible is None else bool(fail_visible)
        # 'sensitive' | 'balanced' | 'strict' — swaps the threat questions for a
        # measured recall/precision trade-off (see SENSITIVITY_MEASURED).
        self.sensitivity = sensitivity if sensitivity in SENSITIVITY_QUESTIONS else "balanced"
        # A confirmed verdict below this confidence is downgraded to rejected — kills
        # the borderline "confirmed at 0.5" noise the CV detectors produce.
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.model = model or self._DEFAULT_MODELS.get(provider, "claude-sonnet-4-6")
        # If the caller left the default Anthropic key env but picked another provider,
        # switch to that provider's conventional env var (e.g. OPENROUTER_API_KEY).
        if api_key_env == "ANTHROPIC_API_KEY" and provider in self.DEFAULT_KEY_ENV:
            api_key_env = self.DEFAULT_KEY_ENV[provider]
        self.api_key_env = api_key_env
        # `local` targets an Ollama server's OpenAI-compatible endpoint by default.
        self.base_url = base_url or ("http://localhost:11434/v1" if provider == "local" else "")
        self.save_dir = Path(save_dir) if save_dir else None
        self._call_count = 0

    @property
    def prompt_version(self) -> str:
        """Short fingerprint of the prompt wording this gate is running."""
        if not getattr(self, "_prompt_version", ""):
            try:
                from cvti.eval.prompt_fingerprint import fingerprint
                self._prompt_version = fingerprint()[:12]
            except Exception:  # noqa: BLE001 - versioning must never break verification
                log.debug("prompt fingerprint unavailable", exc_info=True)
                self._prompt_version = ""
        return self._prompt_version

    def verify(
        self,
        frame: Any,               # a single numpy BGR frame OR a list of them (multi-frame)
        alert: CandidateAlert,
        scene_context: dict | None = None,
        examples: list | None = None,   # past operator feedback for this (camera, rule)
    ) -> VerificationResult:
        self._call_count += 1
        context = scene_context or {}
        environment_type = context.get("environment_type", "unknown")
        scene_description = context.get("scene_description", "No scene description available.")

        template = _COT_PROMPT_TEMPLATE if self.cot else _PROMPT_TEMPLATE
        prompt = template.format(
            environment_type=environment_type,
            scene_description=scene_description,
            rule_name=alert.rule_name,
            detector=alert.detector,
            title=alert.title,
            person_id=alert.person_id if alert.person_id is not None else "unknown",
            object_label=alert.object_label or "unknown",
            question=getattr(alert, "question", None) or _build_question(
                alert.rule_name, environment_type, getattr(alert, "detector", ""),
                self.sensitivity),
            priority=alert.priority,
        )
        mem = _format_examples(examples)
        if mem:
            prompt += mem

        frames = frame if isinstance(frame, list) else [frame]
        frames_bytes = [_encode_frame(f) for f in frames]

        try:
            raw_response = self._call_provider(prompt, frames_bytes, alert)
        except UnsupportedProvider:
            raise                       # a config bug, not a transport blip
        except Exception as exc:  # noqa: BLE001 - transport failure is not a verdict
            log.warning("gate transport failed; alert will be surfaced UNVERIFIED", exc_info=True)
            # Ollama down, model not pulled, request timed out. Previously this
            # propagated to the gate pool, which counted an error and dropped the
            # alert — a real fire, silently discarded, with nothing on screen.
            result = VerificationResult(
                confirmed=self.fail_visible, confidence=0.0,
                reason=UNVERIFIED_REASON,
                alert_priority=alert.priority,
                timestamp=datetime.now(timezone.utc).replace(microsecond=0)
                .isoformat().replace("+00:00", "Z"),
                raw_response="",
                error=f"transport: {type(exc).__name__}: {str(exc)[:160]}")
            result.prompt_version = self.prompt_version
            if self.save_dir:
                _save_artifacts(self.save_dir, self._call_count, frames, alert, result, "")
            return result

        result = _parse_response(raw_response, alert.priority)
        if result.errored and self.fail_visible:
            # A parse failure is not a verdict either. Surface it, flagged.
            result = VerificationResult(
                confirmed=True, confidence=0.0, reason=UNVERIFIED_REASON,
                alert_priority=result.alert_priority, timestamp=result.timestamp,
                raw_response=result.raw_response, error=result.error)

        # Hardening: a confirmed verdict the VLM isn't confident about is not good
        # enough. An unverified alert is exempt — it has no confidence to fall
        # below, and downgrading it would re-hide exactly what we just surfaced.
        if result.confirmed and not result.errored and result.confidence < self.min_confidence:
            result = VerificationResult(
                confirmed=False, confidence=result.confidence,
                reason=f"Below confidence floor ({result.confidence:.2f} < {self.min_confidence:.2f}): {result.reason}",
                alert_priority=result.alert_priority, timestamp=result.timestamp,
                raw_response=result.raw_response)

        result.prompt_version = self.prompt_version
        if self.save_dir:
            _save_artifacts(self.save_dir, self._call_count, frames, alert, result, raw_response)

        return result

    def _call_provider(self, prompt: str, frames_bytes: list[bytes], alert: Any) -> str:
        if self.provider == "mock":
            raw_response = _mock_response(alert)
        elif self.provider == "anthropic":
            raw_response = _call_anthropic(prompt, frames_bytes, self.model, self.api_key_env)
        elif self.provider in ("local", "openai_compatible"):
            raw_response = _call_openai_compatible(
                prompt=prompt,
                frame_bytes=frames_bytes[0],
                model=self.model,
                base_url=self.base_url or "https://api.openai.com/v1",
                # Local Ollama needs no key; OpenAI-compatible clouds do.
                api_key_env="" if self.provider == "local" else self.api_key_env,
            )
        elif self.provider == "openrouter":
            raw_response = _call_openrouter(prompt, frames_bytes, self.model, self.api_key_env)
        elif self.provider == "ollama":
            raw_response = _call_ollama(prompt, frames_bytes, self.model, self.api_key_env,
                                        base_url=self.base_url)
        else:
            raise UnsupportedProvider(f"Unsupported provider: {self.provider}")

        return raw_response


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


def _call_ollama(prompt: str, frames_bytes: list[bytes], model: str, api_key_env: str,
                 base_url: str = "") -> str:
    """Verify via a LOCAL Ollama server — offline, on-device (the edge gate path).

    Ollama exposes an OpenAI-compatible API at localhost:11434 and ignores auth, so we
    reuse the same call path and just point the base URL at it. Set OLLAMA_API_KEY to any
    non-empty value (done automatically here) since the shared helper requires a key.
    """
    os.environ.setdefault(api_key_env, "ollama")
    from cvti.scene.agent_mapper import call_openai_compatible  # local import: only when used
    return call_openai_compatible(
        prompt=prompt,
        frame_bytes=frames_bytes,
        model=model,
        api_key_env=api_key_env,
        # --gate-base-url used to be silently ignored on this (default) path:
        # a nonstandard Ollama host made every alert UNVERIFIED with nothing
        # saying the flag never took. (Audit 23 Aug, #4.)
        api_base_url=base_url or "http://localhost:11434/v1",
    )


def _call_openrouter(prompt: str, frames_bytes: list[bytes], model: str, api_key_env: str) -> str:
    """Verify via an OpenAI-compatible endpoint (OpenRouter / Gemma 4 etc.), one or many frames.

    Reuses agent_mapper.call_openai_compatible — the same retry/backoff path already
    validated against OpenRouter's free tier — so we don't duplicate the HTTP logic.
    """
    from cvti.scene.agent_mapper import call_openai_compatible  # local import: only when used
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


def _call_openai_compatible(
    prompt: str,
    frame_bytes: bytes,
    model: str,
    base_url: str,
    api_key_env: str = "",
) -> str:
    """Call any OpenAI-compatible chat/vision endpoint.

    Used for two cases:
    - provider="local": an Ollama server (http://localhost:11434/v1), no API key.
    - provider="openai_compatible": OpenAI / OpenRouter / etc., key from api_key_env.
    """
    api_key = os.environ.get(api_key_env, "").strip() if api_key_env else ""
    image_url = f"data:image/jpeg;base64,{base64.b64encode(frame_bytes).decode('ascii')}"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    req = urlrequest.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"VLM request failed HTTP {exc.code}: {detail}") from exc
    except urlerror.URLError as exc:
        hint = ""
        if "localhost:11434" in base_url:
            hint = " Is Ollama running? Start it with `ollama serve` and `ollama pull <model>`."
        raise RuntimeError(f"VLM request failed: {exc}.{hint}") from exc

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError("VLM response contained no choices.")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "\n".join(item.get("text", "") for item in content if item.get("type") == "text")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("VLM response contained no text content.")
    return content.strip()


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
        log.warning("gate response unparseable; no verdict reached", exc_info=True)
        # NOT a rejection. The model did not render a verdict — we could not read
        # one. `error` is what tells the live path to surface this to a human
        # instead of dropping it; see FAIL_VISIBLE.
        return VerificationResult(
            confirmed=False,
            confidence=0.0,
            reason=f"TrueSight could not decide: {exc}",
            alert_priority=fallback_priority,
            timestamp=timestamp,
            raw_response=raw,
            error=f"parse: {str(exc)[:160]}",
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
    # Cap the archive: one frame-bearing directory PER VERDICT, outside the
    # retention manager's reach, grew the disk forever on a 24/7 site. Keep
    # the newest 200 — debugging depth without unbounded growth.
    # (RAM/disk audit 24 Aug, service #5.)
    if call_count % 50 == 0:
        try:
            dirs = sorted(d for d in save_dir.glob("gate_*") if d.is_dir())
            for old_dir in dirs[:-200]:
                import shutil
                shutil.rmtree(old_dir, ignore_errors=True)
        except OSError:
            log.debug("gate artifact prune failed", exc_info=True)
    for stale_frame in out_dir.glob("frame*.jpg"):
        stale_frame.unlink()
    for i, fr in enumerate(frames):
        cv2.imwrite(str(out_dir / (f"frame_{i}.jpg" if len(frames) > 1 else "frame.jpg")), fr)
    (out_dir / "alert.json").write_text(
        json.dumps(alert.to_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "verification.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
