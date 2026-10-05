import pytest
from test_concealment import frame
from cvti.retail.concealment import ConcealmentDetector


@pytest.mark.parametrize("fps", [2, 4, 8, 15, 30])
def test_reach_to_waist_survives_sampling_changes(fps):
    detector = ConcealmentDetector()
    proposals = []
    for i in range(fps * 3):
        t = i / fps
        wrist = (200., 110.) if t < .5 else (105., 245.)
        proposals += [a for a in detector.update([frame(t, wrist)], t) if a.candidate]
    assert len(proposals) == 1


@pytest.mark.parametrize("fps", [2, 8, 30])
def test_long_hand_rest_is_not_concealment(fps):
    detector = ConcealmentDetector()
    for i in range(fps * 6):
        t = i / fps
        assert not detector.update([frame(t, (105., 245.))], t)[0].candidate


def test_rewind_clears_previous_reach_and_bag_state():
    detector = ConcealmentDetector()
    detector.update([frame(10., (200., 110.))], 10.)
    detector.update([frame(0., (105., 245.))], 0.)
    assert len(detector._buffers[1]) == 1
    assert detector._buffers[1][0].timestamp == 0.
    assert detector._over_threshold[1] == 0


def test_duplicate_timestamp_cannot_accumulate_dwell():
    detector = ConcealmentDetector()
    for _ in range(30):
        detector.update([frame(0., (105., 245.))], 0.)
    assert len(detector._buffers[1]) == 1
