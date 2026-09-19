from cvti.object_watch.candidate_tracks import CandidateTrackAssociator
from cvti.object_watch.matcher import ObjectCandidate
from cvti.object_watch.presence import PresenceDebouncer


def candidate(box, *, track_id=None, confidence=.9):
    return ObjectCandidate(box, "object", confidence, track_id=track_id)


def test_short_dropout_keeps_id_and_emits_one_presence_token():
    tracker = CandidateTrackAssociator(
        camera_id="cam", source_generation=1, expected_fps=5, max_tracks=8,
    )
    presence = PresenceDebouncer(min_observations=2)
    emitted = []

    first = tracker.assign((candidate((0, 0, 20, 20)),), 0.0)[0]
    token = presence.ready("cam", 1, "recognition", "target", first.track_id, 0, 0.0)
    if token and presence.commit(token): emitted.append(token)
    assert tracker.assign((), .2) == ()
    second = tracker.assign((candidate((1, 0, 21, 20)),), .4)[0]
    token = presence.ready("cam", 1, "recognition", "target", second.track_id, 2, .4)
    if token and presence.commit(token): emitted.append(token)
    third = tracker.assign((candidate((2, 0, 22, 20)),), .6)[0]
    token = presence.ready("cam", 1, "recognition", "target", third.track_id, 3, .6)
    if token and presence.commit(token): emitted.append(token)

    assert first.track_id == second.track_id == third.track_id
    assert len(emitted) == 1


def test_long_gap_expires_and_creates_new_id():
    tracker = CandidateTrackAssociator(
        camera_id="cam", source_generation=1, expected_fps=5, max_tracks=8,
    )
    first = tracker.assign((candidate((0, 0, 20, 20)),), 0.0)[0]
    tracker.assign((), 2.1)
    reacquiring = tracker.assign((candidate((0, 0, 20, 20)),), 2.2)[0]
    second = tracker.assign((candidate((0, 0, 20, 20)),), 2.4)[0]
    assert first.track_id is not None
    assert reacquiring.track_id is None
    assert second.track_id is not None
    assert second.track_id != first.track_id


def test_simultaneous_candidates_are_one_to_one():
    tracker = CandidateTrackAssociator(
        camera_id="cam", source_generation=1, expected_fps=5, max_tracks=8,
    )
    rows = tracker.assign((
        candidate((0, 0, 20, 20)), candidate((40, 0, 60, 20)),
    ), 0.0)
    assert rows[0].track_id is not None
    assert rows[1].track_id is not None
    assert rows[0].track_id != rows[1].track_id


def test_inherited_ids_are_preserved_and_local_namespace_does_not_collide():
    tracker = CandidateTrackAssociator(
        camera_id="cam", source_generation=1, expected_fps=5, max_tracks=8,
    )
    inherited = -(1 << 62)
    rows = tracker.assign((
        candidate((0, 0, 20, 20), track_id=7),
        candidate((40, 0, 60, 20), track_id=inherited),
        candidate((80, 0, 100, 20)),
    ), 0.0)
    assert rows[0].track_id == 7
    assert rows[1].track_id == inherited
    assert rows[2].track_id not in {None, 7, inherited}


def test_low_quality_unassociated_candidate_gets_no_fallback_id():
    tracker = CandidateTrackAssociator(
        camera_id="cam", source_generation=1, expected_fps=5, max_tracks=8,
    )
    good = tracker.assign((candidate((0, 0, 20, 20)),), 0.0)[0]
    poor = tracker.assign((candidate((0, 0, 20, 20), confidence=0.01),), .2)[0]
    assert good.track_id is not None
    assert poor.track_id is None
