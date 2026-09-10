from drones import config
from drones.control.safety import CEILING_SAMPLES, CeilingMonitor

LOW = config.CEILING_DISTANCE / 2


def test_needs_consecutive_readings_before_confirming():
    monitor = CeilingMonitor()
    results = [monitor.update(LOW) for _ in range(CEILING_SAMPLES)]
    assert results == [False] * (CEILING_SAMPLES - 1) + [True]


def test_one_clear_reading_resets_the_count():
    monitor = CeilingMonitor()
    for _ in range(CEILING_SAMPLES - 1):
        monitor.update(LOW)
    assert not monitor.update(None)
    assert not monitor.update(LOW)


def test_out_of_range_and_far_readings_are_not_a_ceiling():
    monitor = CeilingMonitor()
    assert not any(monitor.update(value) for value in [None, 5.0] * 5)
