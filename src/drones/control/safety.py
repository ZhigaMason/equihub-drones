"""Safety checks shared by every way of flying: real, simulated or learned."""
from drones.control.avoidance import ceiling_detected

# Consecutive ceiling readings needed before landing, so that a single
# spurious measurement cannot end the flight.
CEILING_SAMPLES = 3


class CeilingMonitor:
    """Debounced ceiling detection. Call `update()` once per control step."""

    def __init__(self, samples=CEILING_SAMPLES):
        self._samples = samples
        self._hits = 0

    def update(self, up):
        """Feed the upward ranger reading; True once a ceiling is confirmed."""
        if ceiling_detected(up):
            self._hits += 1
        else:
            self._hits = 0
        return self._hits >= self._samples
