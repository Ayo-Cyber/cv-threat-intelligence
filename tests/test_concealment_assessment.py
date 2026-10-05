import pytest
from cvti.verification.concealment_assessment import assess_concealment


def observation(**overrides):
    return dict(item_visible=True, action="insertion", destination="clothing",
                same_subject=True, opening_visible=False, start_frame=1,
                end_frame=3, limitation="none") | overrides


@pytest.mark.parametrize("destination,phrase", [
    ("clothing", "beneath clothing"), ("waistband", "into the waistband"),
    ("personal_bag", "inside a carried bag"), ("pocket", "into a pocket"),
])
def test_destinations(destination, phrase):
    supported, text = assess_concealment(observation(destination=destination, opening_visible=True), 3)
    assert supported and phrase in text
    assert "appears" in text


@pytest.mark.parametrize("changes", [
    {"same_subject": False}, {"same_subject": "true"}, {"item_visible": False},
    {"start_frame": 0}, {"start_frame": True}, {"end_frame": 4}, {"end_frame": 1},
    {"end_frame": "3"}, {"destination": []}, {"destination": "store_basket"},
    {"limitation": "occluded"}, {"limitation": "low_light"}, {"limitation": None},
    {"action": "removal"}, {"action": "holding"}, {"action": "touching"},
    {"destination": "pocket", "opening_visible": False},
])
def test_unsupported_observations_reject(changes):
    assert not assess_concealment(observation(**changes), 3)[0]


def test_crop_is_not_a_later_time_step_and_missing_fields_reject():
    assert not assess_concealment(observation(), 1)[0]
    assert not assess_concealment({}, 3)[0]


def test_limitation_and_normal_action_have_specific_descriptions():
    assert "obscured" in assess_concealment(observation(limitation="occluded"), 3)[1]
    assert "retrieve" in assess_concealment(observation(action="removal"), 3)[1]
