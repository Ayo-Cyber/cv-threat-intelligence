"""PPE compliance: per-person, three-state, policy-driven (16 Sep).

The design rules under test:
  * a required item confidently ABSENT = violation; every item PRESENT =
    compliant; anything else = UNABLE — never a percentage
  * the customer's zone policy decides what is required, not the model
  * an item counts only on the RIGHT person's RIGHT body region; a helmet in
    a hand is not worn; two overlapping people = ownership unclear = unknown
  * a region that cannot be seen (feet below the frame) is unknown, not absent
  * one bad frame never alerts; evidence expires
  * the worker alerts once per person per missing set and reports coverage
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.ppe.assess import (ABSENT, COMPLIANT, PRESENT, UNABLE, UNKNOWN, VIOLATION,
                             TrackEvidence, assess_compliance, observe_people, region_visible)
from cvti.ppe.policy import load_ppe_policy
from cvti.ppe.scanner import PPEScanner

FRAME_HW = (720, 1280)


def _policy(**ppe):
    cam = {"id": "bay", "ppe": ppe or {"required": ["helmet", "vest"]}}
    return load_ppe_policy(cam)


def _person(tid, x=400, y=100, w=120, h=400):
    return (tid, x, y, x + w, y + h)


def _det(phrase, cx, cy, score=0.8, size=40):
    return {"phrase": phrase, "score": score,
            "box": (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)}


# --- policy ------------------------------------------------------------------

def test_policy_zone_union_and_explicit_none():
    pol = _policy(required=["helmet"],
                  zones={"loading_bay": ["helmet", "vest", "boots"], "office": []})
    assert pol.required_for(("loading_bay",)) == ("helmet", "vest", "boots")
    assert pol.required_for(("office",)) == ()             # explicitly nothing
    assert pol.required_for(()) == ("helmet",)              # camera-wide default
    assert pol.required_for(("office", "loading_bay")) == ("helmet", "vest", "boots")


def test_policy_rejects_unknown_item_and_supports_site_items():
    import pytest
    with pytest.raises(ValueError):
        load_ppe_policy({"id": "x", "ppe": {"required": ["helmett"]}})
    pol = load_ppe_policy({"id": "x", "ppe": {
        "required": ["blue_uniform"],
        "items": {"blue_uniform": {"region": "torso", "phrases": ["blue work uniform"]}}}})
    assert pol.items["blue_uniform"].region == "torso"
    assert "blue work uniform" in pol.phrases()
    assert load_ppe_policy({"id": "x"}) is None
    assert load_ppe_policy({"id": "x", "ppe": {"required": []}}) is None


# --- one-frame observation: association + visibility -------------------------

def test_helmet_on_head_is_present_helmet_in_hand_is_not():
    pol = _policy(required=["helmet"])
    p = _person(1)                                  # box x 400..520, y 100..500
    on_head = _det("hard hat", cx=460, cy=130)      # top 22% of the box
    in_hand = _det("hard hat", cx=460, cy=350)      # waist height
    worn = observe_people([p], [on_head], pol, FRAME_HW)[0].items["helmet"]
    carried = observe_people([p], [in_hand], pol, FRAME_HW)[0].items["helmet"]
    assert worn.status == PRESENT and worn.score == 0.8
    assert carried.status == ABSENT


def test_neighbours_vest_does_not_satisfy_this_person():
    pol = _policy(required=["vest"])
    a = _person(1, x=100)
    b = _person(2, x=700)
    vest_on_b = _det("high visibility vest", cx=760, cy=260)
    obs = {o.track_id: o for o in observe_people([a, b], [vest_on_b], pol, FRAME_HW)}
    assert obs[2].items["vest"].status == PRESENT
    assert obs[1].items["vest"].status == ABSENT


def test_overlapping_people_make_ownership_unknown():
    pol = _policy(required=["vest"])
    a = _person(1, x=400)
    b = _person(2, x=440)                           # heavily overlapping torsos
    vest = _det("high visibility vest", cx=470, cy=260)
    obs = {o.track_id: o for o in observe_people([a, b], [vest], pol, FRAME_HW)}
    assert obs[1].items["vest"].status == UNKNOWN
    assert obs[2].items["vest"].status == UNKNOWN
    assert "ownership" in obs[1].items["vest"].reason


def test_feet_below_frame_are_unknown_not_missing():
    pol = _policy(required=["boots"])
    clipped = _person(1, y=400, h=320)              # bottom at 720 = frame edge
    obs = observe_people([clipped], [], pol, FRAME_HW)[0]
    assert obs.items["boots"].status == UNKNOWN
    assert "cut off" in obs.items["boots"].reason
    ok, _ = region_visible((400, 100, 520, 500), "feet", FRAME_HW, 80)
    assert ok


def test_tiny_person_and_missing_detector_are_unknown():
    pol = _policy(required=["helmet"])
    small = _person(1, h=50)
    assert observe_people([small], [], pol, FRAME_HW)[0].items["helmet"].status == UNKNOWN
    down = observe_people([_person(1)], None, pol, FRAME_HW)[0].items["helmet"]
    assert down.status == UNKNOWN and "detector" in down.reason


def test_zero_shot_silence_on_low_recall_items_is_shadow_not_absent():
    # Measured 16 Sep: the open-vocab model missed hi-vis vests on 45/70 vested
    # torsos. Its silence on such items is unknown (shadow), never a violation;
    # a hard hat (0.6-0.8 when worn) stays live; the site can opt vest in.
    pol = _policy(required=["helmet", "vest"])
    obs = observe_people([_person(1)], [], pol, FRAME_HW, zero_shot=True)[0]
    assert obs.items["helmet"].status == ABSENT
    assert obs.items["vest"].status == UNKNOWN and "shadow" in obs.items["vest"].reason
    assert pol.shadow_items(zero_shot=True) == ("vest",)
    assert pol.shadow_items(zero_shot=False) == ()            # a trained detector: live
    trusted = _policy(required=["vest"], trust_absent=["vest"])
    assert observe_people([_person(1)], [], trusted, FRAME_HW, zero_shot=True)[0] \
        .items["vest"].status == ABSENT
    # a vest actually seen is PRESENT either way
    seen = observe_people([_person(1)], [_det("high visibility vest", 460, 260, 0.4)],
                          pol, FRAME_HW, zero_shot=True)[0]
    assert seen.items["vest"].status == PRESENT


def test_item_floor_is_per_item():
    pol = _policy(required=["helmet", "vest"])
    p = _person(1)
    weak_helmet = _det("hard hat", cx=460, cy=130, score=0.30)       # below helmet 0.35
    weak_vest = _det("high visibility vest", cx=460, cy=260, score=0.30)  # above vest 0.20
    obs = observe_people([p], [weak_helmet, weak_vest], pol, FRAME_HW)[0]
    assert obs.items["helmet"].status == ABSENT
    assert obs.items["vest"].status == PRESENT


# --- evidence over time -----------------------------------------------------------

def _obs(status, item="helmet"):
    from cvti.ppe.assess import ItemObservation, PersonObservation
    o = PersonObservation(1, (0, 0, 10, 10), (), (item,))
    o.items[item] = ItemObservation(item, status, 0.7 if status == PRESENT else 0.0, "r")
    return o


def test_one_bad_frame_never_decides_and_evidence_expires():
    ev = TrackEvidence(window_seconds=8.0, confirm=3)
    ev.add(_obs(ABSENT), 0.0)
    assert ev.verdict("helmet", 0.0)[0] == UNKNOWN          # 1 of 3
    ev.add(_obs(ABSENT), 1.0)
    ev.add(_obs(ABSENT), 2.0)
    assert ev.verdict("helmet", 2.0)[0] == ABSENT           # unanimous
    ev.add(_obs(PRESENT), 3.0)                              # put it on
    assert ev.verdict("helmet", 3.0)[0] == UNKNOWN          # last 3: A A P -> inconsistent
    ev.add(_obs(PRESENT), 4.0)
    assert ev.verdict("helmet", 4.0)[0] == PRESENT          # majority present
    assert ev.verdict("helmet", 20.0)[0] == UNKNOWN         # everything expired


def test_unknown_frames_do_not_count_as_evidence():
    ev = TrackEvidence(8.0, 3)
    for t in range(5):
        ev.add(_obs(UNKNOWN), float(t))
    status, why = ev.verdict("helmet", 5.0)
    assert status == UNKNOWN and why == "r"                 # carries the reason


# --- the decision -------------------------------------------------------------------

def test_two_of_three_is_not_67_percent_compliant():
    req = ("helmet", "vest", "boots")
    v = {"helmet": (PRESENT, ""), "vest": (PRESENT, ""), "boots": (UNKNOWN, "feet cut off")}
    c = assess_compliance(req, v)
    assert c.status == UNABLE and c.missing == () and c.present == ("helmet", "vest")
    assert c.unknown == {"boots": "feet cut off"}
    v["boots"] = (ABSENT, "absent in 3 checks")
    assert assess_compliance(req, v).status == VIOLATION
    v["boots"] = (PRESENT, "")
    assert assess_compliance(req, v).status == COMPLIANT
    assert assess_compliance((), {}).status == "not_required"


def test_missing_helmet_is_violation_even_with_vest_and_boots():
    v = {"helmet": (ABSENT, ""), "vest": (PRESENT, ""), "boots": (PRESENT, "")}
    c = assess_compliance(("helmet", "vest", "boots"), v)
    assert c.status == VIOLATION and c.missing == ("helmet",)
    assert c.summary().startswith("missing helmet")


# --- the worker -----------------------------------------------------------------------

class _Sink:
    def __init__(self):
        self.alerts = []

    def handle(self, alert, result):
        self.alerts.append((alert, result))


class _Det:
    """A stand-in detector: returns whatever the test scripts per call."""

    def __init__(self):
        self.script = []
        self.calls = 0

    def detect(self, frame, phrases, floor=None):
        self.calls += 1
        return self.script.pop(0) if self.script else []

    zero_shot = False       # stands in for a trained detector: absence is live

    def status(self):
        return {"loaded": True, "weights": "fake.pt"}


def _scanner(sink, det, people, zones=None, ppe=None):
    cam = {"id": "bay", "ppe": ppe or {"required": ["helmet"], "confirm_observations": 3,
                                       "clear_after_seconds": 5.0}}
    frame = np.zeros((*FRAME_HW, 3), dtype=np.uint8)
    return PPEScanner([cam], sink, frame_source=lambda c: frame,
                      boxes_source=lambda c: people, zones_source=lambda c: zones or {},
                      detector=det, clock=lambda: 0.0)


def test_worker_alerts_once_per_person_after_confirmation():
    sink, det = _Sink(), _Det()
    s = _scanner(sink, det, [_person(7)])
    for t in range(4):
        s.step("bay", float(t))                          # no helmet seen, 4 cycles
    assert len(sink.alerts) == 1                          # confirmed at cycle 3, alerted once
    alert, result = sink.alerts[0]
    assert alert.rule_name == "ppe:camera" and alert.track_id == 7
    assert alert.title == "PPE: MISSING HARD HAT" and result.confirmed
    assert alert.payload["ppe"]["status"] == VIOLATION
    assert 0.5 < result.confidence < 0.95                 # never certain
    st = s.status()["cameras"]["bay"]
    assert st["alerts"] == 1 and st["violations"] >= 1 and st["open_violations"] == 1


def test_worker_never_alerts_when_unable_and_reports_coverage():
    sink, det = _Sink(), _Det()
    clipped = _person(3, y=400, h=320)                    # feet below the frame
    s = _scanner(sink, det, [clipped], ppe={"required": ["boots"]})
    for t in range(5):
        s.step("bay", float(t))
    assert sink.alerts == []
    st = s.status()["cameras"]["bay"]
    assert st["unable"] == 5 and st["assessed"] == 0 and st["unable_rate"] == 1.0
    assert "feet cut off by frame edge" in st["unknown_reasons"]


def test_worker_with_zero_shot_detector_shadows_vest_but_alerts_helmet():
    sink, det = _Sink(), _Det()
    det.zero_shot = True
    s = _scanner(sink, det, [_person(1)], ppe={"required": ["helmet", "vest"],
                                               "confirm_observations": 2})
    s.step("bay", 0.0)
    s.step("bay", 1.0)
    assert len(sink.alerts) == 1
    assert sink.alerts[0][0].title == "PPE: MISSING HARD HAT"      # vest not claimed
    assert sink.alerts[0][0].payload["ppe"]["unassessable"] == ["vest"]   # reported, never decided on
    assert s.status()["cameras"]["bay"]["shadow_items"] == ["vest"]


def test_worker_uses_zone_policy_and_closes_on_compliance():
    sink, det = _Sink(), _Det()
    people = [_person(1)]
    # zone policy: helmet required only in the loading bay
    s = _scanner(sink, det, people, zones={1: ("loading_bay",)},
                 ppe={"zones": {"loading_bay": ["helmet"]}, "confirm_observations": 2})
    s.step("bay", 0.0)
    s.step("bay", 1.0)
    assert len(sink.alerts) == 1 and sink.alerts[0][0].zone == "loading_bay"
    assert sink.alerts[0][0].rule_name == "ppe:loading_bay"
    # helmet goes on: two present observations -> compliant -> incident closes
    det.script = [[_det("hard hat", 460, 130)], [_det("hard hat", 460, 130)]]
    s.step("bay", 2.0)
    s.step("bay", 3.0)
    assert s.open_violations == {}
    assert len(sink.alerts) == 1                          # no 'compliant' alert, ever


def test_worker_does_not_touch_people_outside_policy_zones():
    sink, det = _Sink(), _Det()
    s = _scanner(sink, det, [_person(1)], zones={1: ("office",)},
                 ppe={"zones": {"office": [], "bay": ["helmet"]}})
    for t in range(4):
        s.step("bay", float(t))
    assert sink.alerts == [] and det.calls == 4           # looked, required nothing
    assert s.status()["cameras"]["bay"]["not_required"] == 4


def test_small_head_is_unknown_not_bare():
    # 16 Sep, industrial clip: workers ~70 px tall in hard hats were called
    # bare-headed — a 15 px head band is below what the detector resolves.
    # Too small to judge is UNKNOWN, never absent.
    pol = _policy(required=["helmet"])
    short = _person(1, h=70)                        # head band = 0.22 * 70 ≈ 15 px
    obs = observe_people([short], [], pol, FRAME_HW)[0]
    assert obs.items["helmet"].status == UNKNOWN and "too small" in obs.items["helmet"].reason
    tall = _person(2, h=120)                        # head band ≈ 26 px: judgeable
    assert observe_people([tall], [], pol, FRAME_HW)[0].items["helmet"].status == ABSENT


def test_long_garment_counts_as_worn_on_torso_and_carried_helmet_still_does_not():
    # 16 Sep, lab clips: a lab coat's box spans shoulders to knees, so its
    # centre sat just below the torso band and worn coats read as ABSENT.
    # Association is now band OVERLAP, not centre-in-band.
    pol = load_ppe_policy({"id": "lab", "ppe": {
        "required": ["lab_coat", "helmet"],
        "items": {"lab_coat": {"region": "torso", "phrases": ["lab coat"],
                               "absent_reliable_zero_shot": True}}}})
    p = _person(1)                                   # 400..520 x 100..500
    coat = {"phrase": "lab coat", "score": 0.8, "box": (395, 180, 525, 460)}   # shoulders→knees
    obs = observe_people([p], [coat], pol, FRAME_HW)[0]
    assert obs.items["lab_coat"].status == PRESENT
    waist_helmet = _det("hard hat", cx=460, cy=350)  # carried: no overlap with head band
    assert observe_people([p], [waist_helmet], pol, FRAME_HW)[0].items["helmet"].status == ABSENT
    worn_helmet = _det("hard hat", cx=460, cy=130)
    assert observe_people([p], [worn_helmet], pol, FRAME_HW)[0].items["helmet"].status == PRESENT
    # an item box beside the person (neighbour's coat) is not theirs
    beside = {"phrase": "lab coat", "score": 0.8, "box": (600, 180, 700, 460)}
    assert observe_people([p], [beside], pol, FRAME_HW)[0].items["lab_coat"].status == ABSENT


def test_phrase_shared_with_an_unrequired_catalog_item_goes_to_the_required_one():
    # 'lab coat' is also a phrase of the catalog's `coverall`; when a site
    # defines its own lab_coat and requires it, the detection is theirs.
    pol = load_ppe_policy({"id": "lab", "ppe": {
        "required": ["lab_coat"],
        "items": {"lab_coat": {"region": "torso", "phrases": ["lab coat"]}}}})
    assert pol.item_for_phrase("lab coat").key == "lab_coat"
    assert pol.item_for_phrase("hard hat") is None          # helmet not required here


def test_waist_up_closeup_cannot_judge_the_torso():
    # 16 Sep, lab clips: an interview close-up ends at the frame edge and is
    # nearly as wide as tall — its "torso band" is the neck. A coat below the
    # frame is not a missing coat.
    pol = _policy(required=["vest"])
    closeup = (1, 300, 300, 700, 720)                # 400 wide x 420 tall, bottom at frame edge
    obs = observe_people([closeup], [], pol, FRAME_HW, zero_shot=False)[0]
    assert obs.items["vest"].status == UNKNOWN and "cut off" in obs.items["vest"].reason
    standing = (2, 400, 100, 520, 720)               # 120 wide x 620 tall: a whole person
    assert observe_people([standing], [], pol, FRAME_HW, zero_shot=False)[0].items["vest"].status == ABSENT


def test_shadow_items_do_not_block_a_verdict_on_the_judged_ones():
    # A lab policy with goggles+gloves in shadow must still be able to say
    # "lab coat: compliant" — otherwise everyone is 'unable' forever and the
    # one thing the detector DOES know is hidden.
    req = ("lab_coat", "goggles", "gloves")
    v = {"lab_coat": (PRESENT, "seen 3/3"), "goggles": (UNKNOWN, "shadow"), "gloves": (UNKNOWN, "shadow")}
    c = assess_compliance(req, v, shadow=("goggles", "gloves"))
    assert c.status == COMPLIANT and c.unassessable == ("goggles", "gloves")
    assert "not assessable" in c.summary()
    v["lab_coat"] = (ABSENT, "absent in 3 checks")
    assert assess_compliance(req, v, shadow=("goggles", "gloves")).status == VIOLATION
    # nothing judgeable at all -> unable, never compliant by default
    assert assess_compliance(("goggles",), {"goggles": (UNKNOWN, "shadow")}, shadow=("goggles",)).status == UNABLE


def test_person_half_out_of_the_side_of_the_frame_is_not_judged():
    # 16 Sep, chemistry-lab clip: a student stepping out of frame at the left
    # edge had her coat cut in two — the detector saw neither half and she
    # read as coatless. Torso and hands are judged once fully in view.
    pol = _policy(required=["vest"])
    at_edge = (1, 0, 100, 130, 500)
    obs = observe_people([at_edge], [], pol, FRAME_HW, zero_shot=False)[0]
    assert obs.items["vest"].status == UNKNOWN and "(side)" in obs.items["vest"].reason
    inside = (2, 40, 100, 170, 500)
    assert observe_people([inside], [], pol, FRAME_HW, zero_shot=False)[0].items["vest"].status == ABSENT
    # the head is still judgeable at the edge — a helmet is not cut in two
    helm = _policy(required=["helmet"])
    assert observe_people([at_edge], [], helm, FRAME_HW, zero_shot=False)[0].items["helmet"].status == ABSENT
