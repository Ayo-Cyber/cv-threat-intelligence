from cvti.serving.alert_queue import AlertQueue, QueuedAlert


def alert(camera, timestamp, priority="high"):
    return QueuedAlert(camera_id=camera, rule_name="product_concealment",
                       priority=priority, title="Possible concealment", timestamp=timestamp)


def test_rewinding_short_clip_cannot_overtake_waiting_camera():
    queue = AlertQueue(cooldown_seconds=0)
    wig = alert("retail_1", 18)
    queue.add(wig)
    for i in range(5):
        queue.add(alert(f"short_{i}", 3))
    assert queue.drain(1) == [wig]


def test_critical_still_preempts_and_equal_priority_preserves_arrival():
    queue = AlertQueue(cooldown_seconds=0)
    first, second, urgent = alert("one", 1000), alert("two", 1), alert("fire", 2000, "critical")
    for item in (first, second, urgent):
        queue.add(item)
    assert queue.drain(3) == [urgent, first, second]


def test_capacity_retains_oldest_equal_priority_not_lowest_media_timestamp():
    queue = AlertQueue(cooldown_seconds=0, max_pending=2)
    first, second, newest = alert("one", 100), alert("two", 50), alert("three", 0)
    assert queue.add(first)
    assert queue.add(second)
    assert not queue.add(newest)
    assert queue.drain(2) == [first, second]
