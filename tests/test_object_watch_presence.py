from cvti.object_watch.presence import PresenceDebouncer


def test_presence_requires_distinct_observations_and_commit() -> None:
    subject = PresenceDebouncer(min_observations=2)
    args = ("cam", 1, "rule", "target", 7)
    assert subject.ready(*args, 1, 0.0) is None
    assert subject.ready(*args, 1, 0.1) is None
    token = subject.ready(*args, 2, 0.2)
    assert token is not None
    assert subject.ready(*args, 3, 0.3) is not None
    assert subject.commit(token)
    assert subject.ready(*args, 4, 0.4) is None


def test_unavailable_does_not_rearm_but_observed_gap_does() -> None:
    subject = PresenceDebouncer(min_observations=1, rearm_gap_seconds=1.0)
    args = ("cam", 1, "rule", "target", 7)
    token = subject.ready(*args, 1, 0.0)
    assert token is not None and subject.commit(token)
    assert subject.ready(*args, 2, 4.0, available=False) is None
    assert subject.ready(*args, 3, 4.1) is not None


def test_permanent_track_keeps_bounded_observation_history() -> None:
    subject = PresenceDebouncer(min_observations=2)
    args = ("cam", 1, "rule", "target", 7)
    for sequence in range(10_000):
        subject.ready(*args, sequence, sequence / 10.0)
    state = next(iter(subject._states.values()))
    assert len(state.observations) <= subject.min_observations
