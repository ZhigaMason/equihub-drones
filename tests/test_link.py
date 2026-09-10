import cflib.crtp
import pytest

from drones.control.mixer import Ranges
from drones.crazyflie import link

RADIO = 'radio://0/80/2M/E7E7E7E7E7'


@pytest.fixture
def scan(monkeypatch):
    """Make the next interface scan report exactly these URIs."""
    def found(*uris):
        monkeypatch.setattr(cflib.crtp, 'scan_interfaces',
                            lambda: [(uri, '') for uri in uris])
    return found


def test_auto_picks_the_only_interface(scan):
    scan('usb://0')
    assert link.resolve_uri('auto') == 'usb://0'
    assert link.resolve_uri('AUTO') == 'usb://0'


def test_explicit_uri_that_is_present(scan):
    scan('usb://0', RADIO)
    assert link.resolve_uri(RADIO) == RADIO


def test_explicit_uri_that_is_missing_names_what_was_found(scan):
    scan('usb://0')
    with pytest.raises(RuntimeError, match='Found: usb://0'):
        link.resolve_uri(RADIO)


def test_explicit_radio_is_tried_when_the_scan_is_empty(scan):
    # A switched-off drone never shows up in a radio scan.
    scan()
    assert link.resolve_uri(RADIO) == RADIO


def test_auto_with_nothing_found(scan):
    scan()
    with pytest.raises(RuntimeError, match='No Crazyflie found'):
        link.resolve_uri('auto')


def test_auto_with_several_interfaces_refuses_to_guess(scan):
    scan('usb://0', RADIO)
    with pytest.raises(RuntimeError, match='Several interfaces'):
        link.resolve_uri('auto')


def test_read_ranges_copies_every_ranger():
    class Ranger:
        front, back, left, right, up, down = 1.0, 2.0, None, 0.5, 1.5, 0.3
    assert link.read_ranges(Ranger()) == Ranges(1.0, 2.0, None, 0.5, 1.5, 0.3)
