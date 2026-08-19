"""A stable fingerprint of every prompt the gate uses.

The point is narrow: detect that the wording changed. Not that `gate.py` changed
— reformatting, a new provider, a renamed variable are all irrelevant to the
numbers. Only the text handed to the model matters, because only that moves
precision.

Taken from the imported module, not by parsing the source. Two of these tables
are built by referencing the others (`_DETECTOR_QUESTIONS` reuses entries from
`_QUESTIONS`), so a literal parse silently skips them — and one of the entries
it skipped is the weapons question, on an always-on critical detector. A
fingerprint with a hole in it is worse than none, because it reads as coverage.

Importing `gate.py` costs a cv2 import and nothing else: no torch, no models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

GATE_SOURCE = Path("cvti/verification/gate.py")

# The constants whose text reaches the model. Anything not on this list does not
# change a verdict, so a change to it must not fail CI.
PROMPT_NAMES = (
    "_PROMPT_TEMPLATE",
    "_COT_PROMPT_TEMPLATE",
    "_QUESTIONS",
    "_DETECTOR_QUESTIONS",
    "SENSITIVITY_QUESTIONS",
)


def extract_prompts(module=None) -> dict:
    """The prompt constants, as data, in a deterministic order."""
    if module is None:
        from cvti.verification import gate as module      # noqa: N813
    found: dict = {}
    for name in PROMPT_NAMES:
        value = getattr(module, name, None)
        if value is not None:
            found[name] = value
    return found


def fingerprint(module=None) -> str:
    """SHA-256 over the prompt text alone. Stable across unrelated edits."""
    prompts = extract_prompts(module)
    blob = json.dumps(prompts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def describe(module=None) -> dict:
    prompts = extract_prompts(module)
    return {"fingerprint": fingerprint(module),
            "constants": {name: (len(v) if isinstance(v, (dict, list)) else len(str(v)))
                          for name, v in sorted(prompts.items())},
            "missing": [n for n in PROMPT_NAMES if n not in prompts]}
