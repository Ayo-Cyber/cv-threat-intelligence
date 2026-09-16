"""What the customer requires, where. Configuration — never the model's guess.

A camera opts in with a `ppe` block:

    "ppe": {
        "required": ["helmet", "vest"],              # anywhere on this camera
        "zones": {                                   # per drawn zone (union wins)
            "loading_bay": ["helmet", "vest", "boots"],
            "office_corridor": []                    # explicitly nothing
        },
        "items": {                                   # optional site-specific items
            "blue_uniform": {"region": "torso", "phrases": ["blue work uniform"]}
        },
        "cadence_seconds": 1.0,
        "confirm_observations": 3,
        "window_seconds": 8.0,
        "min_person_height_px": 80,
        "clear_after_seconds": 30.0,
        "priority": "high"
    }

Items live in a catalog keyed by the body REGION they are worn on — that is
what lets the assessor say "helmet on the head" rather than "a helmet is
somewhere in the frame", and what makes a helmet carried in a hand not count.
The catalog is industrial, not construction-only: CHI is food & beverage, so
hair nets, aprons and coveralls sit beside hard hats and hi-vis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REGIONS = ("head", "face", "torso", "hands", "feet")


@dataclass(frozen=True)
class PPEItem:
    key: str
    label: str
    region: str
    phrases: tuple[str, ...]
    # The floor a grounded detection must clear to count as PRESENT. Tuned per
    # item — open-vocab scores for a hi-vis vest and a hard hat are not on the
    # same scale (measured 14-16 Sep: hard hats 0.6-0.8, vests 0.2-0.4).
    min_score: float = 0.30
    # What CCTV cannot establish about this item. Surfaces in the verdict so
    # nobody sells appearance detection as certification.
    visual_only_note: str = ""
    # Can the ZERO-SHOT detector's silence be trusted as absence? Measured 16
    # Sep on yolov8s-worldv2: hard hats score 0.6-0.8 when worn; hi-vis vests
    # went undetected on 45 of 70 vested torsos (recall ~35%). Where recall is
    # that poor, "not seen" is not "not worn" — the item is assessed in SHADOW
    # (counted as unknown, never alerted) until a trained detector answers, or
    # the site opts in with `trust_absent`. Present calls are unaffected.
    absent_reliable_zero_shot: bool = True


CATALOG: dict[str, PPEItem] = {
    "helmet": PPEItem("helmet", "hard hat", "head",
                      ("hard hat", "safety helmet", "construction helmet"), 0.35),
    "hairnet": PPEItem("hairnet", "hair net", "head",
                       ("hair net", "hair cover", "bouffant cap"), 0.30,
                       absent_reliable_zero_shot=False),
    "goggles": PPEItem("goggles", "safety goggles", "face",
                       ("safety goggles", "safety glasses", "protective eyewear"), 0.30,
                       absent_reliable_zero_shot=False),
    "mask": PPEItem("mask", "face mask", "face",
                    ("face mask", "surgical mask", "respirator"), 0.30,
                    absent_reliable_zero_shot=False),
    "vest": PPEItem("vest", "high-visibility vest", "torso",
                    ("high visibility vest", "reflective safety vest", "hi-vis vest"), 0.20,
                    absent_reliable_zero_shot=False),
    "apron": PPEItem("apron", "apron", "torso", ("apron", "work apron"), 0.30),
    "coverall": PPEItem("coverall", "coverall", "torso",
                        ("coverall", "overalls", "lab coat", "protective suit"), 0.30),
    "gloves": PPEItem("gloves", "safety gloves", "hands",
                      ("safety gloves", "work gloves", "rubber gloves"), 0.30,
                      absent_reliable_zero_shot=False),
    "boots": PPEItem("boots", "safety boots", "feet",
                     ("safety boots", "work boots", "rubber boots"), 0.30,
                     "appearance only — cannot establish steel toe, puncture "
                     "resistance or certification", absent_reliable_zero_shot=False),
}


@dataclass
class PPEPolicy:
    camera_id: str
    default_required: tuple[str, ...] = ()
    zone_required: dict[str, tuple[str, ...]] = field(default_factory=dict)
    items: dict[str, PPEItem] = field(default_factory=lambda: dict(CATALOG))
    cadence_seconds: float = 1.0
    confirm_observations: int = 3
    window_seconds: float = 8.0
    min_person_height_px: int = 80
    clear_after_seconds: float = 30.0
    priority: str = "high"
    # Items whose ABSENCE the site accepts from a zero-shot detector anyway.
    trust_absent: tuple[str, ...] = ()

    def shadow_items(self, zero_shot: bool) -> tuple[str, ...]:
        """Required items assessed but never alerted with this detector:
        their 'not seen' cannot be trusted as 'not worn'."""
        if not zero_shot:
            return ()
        return tuple(k for k in self.all_items()
                     if not self.items[k].absent_reliable_zero_shot and k not in self.trust_absent)

    def required_for(self, zones: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
        """The items a person standing in `zones` must wear.

        A person in a zone with a requirement gets the UNION of every zone
        they are in. A zone configured with `[]` says "nothing required
        here" and contributes nothing. A person in no configured zone gets
        the camera-wide default."""
        req: list[str] = []
        matched = False
        for z in zones or ():
            if z in self.zone_required:
                matched = True
                for item in self.zone_required[z]:
                    if item not in req:
                        req.append(item)
        if not matched:
            req = list(self.default_required)
        return tuple(req)

    def phrases(self) -> list[str]:
        """Every phrase the detector must look for, across all requirements."""
        wanted: list[str] = []
        for key in self.all_items():
            for p in self.items[key].phrases:
                if p not in wanted:
                    wanted.append(p)
        return wanted

    def all_items(self) -> tuple[str, ...]:
        keys: list[str] = list(self.default_required)
        for req in self.zone_required.values():
            for k in req:
                if k not in keys:
                    keys.append(k)
        return tuple(keys)

    def item_for_phrase(self, phrase: str) -> PPEItem | None:
        for item in self.items.values():
            if phrase in item.phrases:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": list(self.default_required),
            "zones": {z: list(r) for z, r in self.zone_required.items()},
            "items": {k: {"label": v.label, "region": v.region, "phrases": list(v.phrases),
                          "min_score": v.min_score, "visual_only_note": v.visual_only_note}
                      for k, v in self.items.items() if k in self.all_items()},
            "cadence_seconds": self.cadence_seconds,
            "confirm_observations": self.confirm_observations,
            "window_seconds": self.window_seconds,
            "min_person_height_px": self.min_person_height_px,
            "clear_after_seconds": self.clear_after_seconds,
            "priority": self.priority,
            "trust_absent": list(self.trust_absent),
        }


def _item_from_config(key: str, spec: dict) -> PPEItem:
    region = str(spec.get("region", "torso"))
    if region not in REGIONS:
        raise ValueError(f"ppe item '{key}': region must be one of {REGIONS}, got '{region}'")
    phrases = tuple(str(p) for p in (spec.get("phrases") or [spec.get("label", key)]))
    return PPEItem(key, str(spec.get("label", key.replace("_", " "))), region, phrases,
                   float(spec.get("min_score", 0.30)), str(spec.get("visual_only_note", "")),
                   # A site-defined item has no measurement behind it: shadow
                   # by default, live only when the site says so.
                   bool(spec.get("absent_reliable_zero_shot", False)))


def load_ppe_policy(cam: dict) -> PPEPolicy | None:
    """The policy for one camera dict, or None when it has no `ppe` block.

    Unknown item names are an error at load time — a typo must not become a
    requirement nobody can satisfy, silently marking everyone `unable`."""
    spec = cam.get("ppe")
    if not spec:
        return None
    items = dict(CATALOG)
    for key, item_spec in (spec.get("items") or {}).items():
        items[str(key)] = _item_from_config(str(key), item_spec or {})

    def _req(values: Any, where: str) -> tuple[str, ...]:
        out: list[str] = []
        for v in values or ():
            v = str(v)
            if v not in items:
                raise ValueError(f"ppe {where}: unknown item '{v}' "
                                 f"(known: {sorted(items)})")
            if v not in out:
                out.append(v)
        return tuple(out)

    zones = {str(z): _req(req, f"zone '{z}'") for z, req in (spec.get("zones") or {}).items()}
    policy = PPEPolicy(
        camera_id=str(cam.get("id", "?")),
        default_required=_req(spec.get("required"), "required"),
        zone_required=zones,
        items=items,
        cadence_seconds=max(0.2, float(spec.get("cadence_seconds", 1.0))),
        confirm_observations=max(1, int(spec.get("confirm_observations", 3))),
        window_seconds=max(1.0, float(spec.get("window_seconds", 8.0))),
        min_person_height_px=max(16, int(spec.get("min_person_height_px", 80))),
        clear_after_seconds=max(1.0, float(spec.get("clear_after_seconds", 30.0))),
        priority=str(spec.get("priority", "high")),
        trust_absent=_req(spec.get("trust_absent"), "trust_absent"),
    )
    if not policy.all_items():
        return None          # a ppe block that requires nothing is not a policy
    return policy
