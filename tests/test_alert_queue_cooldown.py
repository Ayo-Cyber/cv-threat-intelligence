from cvti.serving.alert_queue import AlertQueue, QueuedAlert


def alert(timestamp, camera="retail_1", track=1):
    return QueuedAlert(camera_id=camera, rule_name="product_concealment",
                       priority="high", title="Possible product concealment",
                       timestamp=timestamp, track_id=track)


def test_replay_rewind_does_not_suppress_subject_forever():
    now = [0.0]
    queue = AlertQueue(cooldown_seconds=60, clock=lambda: now[0])
    assert queue.add(alert(28))
    now[0] = 10
    assert not queue.add(alert(3))
    now[0] = 60
    assert queue.add(alert(2))


def test_media_jump_cannot_bypass_runtime_cooldown():
    queue = AlertQueue(clock=lambda: 10)
    assert queue.add(alert(0))
    assert not queue.add(alert(1000))


def test_other_people_and_cameras_remain_independent():
    queue = AlertQueue(clock=lambda: 10)
    assert queue.add(alert(0))
    assert queue.add(alert(0, track=2))
    assert queue.add(alert(0, camera="retail_2"))


def test_duplicates_do_not_extend_cooldown():
    now = [0.0]
    queue = AlertQueue(cooldown_seconds=60, clock=lambda: now[0])
    assert queue.add(alert(0))
    now[0] = 59
    assert not queue.add(alert(2))
    now[0] = 60
    assert queue.add(alert(3))
