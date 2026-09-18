"""The AI-deck stream reader, fed over a real socket the bytes the deck's streamer sends."""
import importlib.util
import logging
import socket
import struct
import time

import numpy as np
import pytest

from drones.crazyflie import camera

# Decoding needs the `camera` extra; reading the stream does not, so only those tests skip.
needs_cv2 = pytest.mark.skipif(importlib.util.find_spec('cv2') is None,
                               reason='needs OpenCV, from the camera extra')


def cpx(payload, routing=0x09, function=0x05):
    return struct.pack('<HBB', len(payload) + 2, routing, function) + payload


def image(data, width=4, height=2, fmt=camera.FORMAT_RAW, chunk=3):
    """A frame as the deck sends it: a header packet, then the pixels in `chunk`-byte packets."""
    header = camera.IMAGE_HEADER.pack(camera.IMAGE_MAGIC, width, height, 1, fmt, len(data))
    return cpx(header) + b''.join(cpx(data[i:i + chunk]) for i in range(0, len(data), chunk))


@pytest.fixture
def pipe():
    deck, viewer = socket.socketpair()
    viewer.settimeout(2.0)
    yield deck, viewer
    deck.close()
    viewer.close()


def wait_for(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)


def test_a_frame_split_over_packets_is_reassembled(pipe):
    deck, viewer = pipe
    deck.sendall(image(bytes(range(8))))
    frame = camera.read_frame(viewer)
    assert (frame.width, frame.height, frame.format) == (4, 2, camera.FORMAT_RAW)
    assert frame.data == bytes(range(8))


def test_packets_before_an_image_header_are_skipped(pipe):
    # Connecting mid-frame lands among that frame's pixel packets.
    deck, viewer = pipe
    deck.sendall(cpx(b'\xbc' + bytes(5)) + cpx(bytes(11)) + image(b'abcdefgh'))
    assert camera.read_frame(viewer).data == b'abcdefgh'


def test_a_closed_connection_is_an_error_not_a_hang(pipe):
    deck, viewer = pipe
    deck.sendall(image(bytes(8))[:20])
    deck.close()
    with pytest.raises(ConnectionError):
        camera.read_frame(viewer)


def test_the_stream_keeps_only_the_newest_frame_and_reports_why_it_stopped(pipe):
    deck, viewer = pipe
    stream = camera.FrameStream(viewer).start()
    for n in range(3):
        deck.sendall(image(bytes([n]) * 8))
    wait_for(lambda: stream.latest()[0] == 3)
    assert stream.latest() == (3, camera.Frame(4, 2, 1, camera.FORMAT_RAW, bytes([2]) * 8))

    deck.close()

    def stopped():
        try:
            stream.latest()
        except ConnectionError:
            return True
        return False
    wait_for(stopped)
    assert stopped()
    stream.close()


def test_closing_the_stream_is_not_an_error(pipe):
    _deck, viewer = pipe
    stream = camera.FrameStream(viewer).start()
    stream.close()
    assert stream.latest() == (0, None)


def test_an_unreachable_deck_names_the_setting_to_check():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with pytest.raises(RuntimeError, match='AIDECK_HOST'):
        camera.connect('127.0.0.1', port, timeout=1.0)


@pytest.fixture
def deck():
    """A stand-in for the deck's streamer: a listening socket on localhost."""
    with socket.create_server(('127.0.0.1', 0)) as server:
        server.settimeout(5.0)
        yield server


def test_the_feed_passes_jpeg_frames_through_untouched(deck):
    feed = camera.CameraFeed(*deck.getsockname(), retry=0.05).start()
    try:
        conn, _ = deck.accept()
        with conn:
            wait_for(lambda: feed.status() == 'waiting for frames')
            assert feed.status() == 'waiting for frames'
            conn.sendall(image(b'jpeg-bytes', fmt=1))
            wait_for(lambda: feed.latest()[0] == 1)
            assert feed.latest() == (1, b'jpeg-bytes')
            assert feed.status() == 'live'
    finally:
        feed.stop()


def test_the_feed_reconnects_after_the_deck_drops(deck):
    feed = camera.CameraFeed(*deck.getsockname(), retry=0.5).start()
    try:
        conn, _ = deck.accept()
        conn.close()
        wait_for(lambda: feed.status().startswith('no video'))
        assert 'closed' in feed.status()
        conn, _ = deck.accept()
        with conn:
            conn.sendall(image(b'again', fmt=1))
            wait_for(lambda: feed.latest()[0] == 1)
            assert feed.latest() == (1, b'again')
    finally:
        feed.stop()


def test_an_unreachable_deck_is_a_status_not_a_crash():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    feed = camera.CameraFeed('127.0.0.1', port, retry=0.05).start()
    try:
        wait_for(lambda: feed.status().startswith('no video'))
        assert feed.status().startswith('no video: Cannot reach')
        assert feed.latest() == (0, None)
    finally:
        feed.stop()


def test_stopping_the_feed_does_not_wait_for_a_silent_deck(deck):
    feed = camera.CameraFeed(*deck.getsockname(), retry=0.05).start()
    conn, _ = deck.accept()
    with conn:
        wait_for(lambda: feed.status() == 'waiting for frames')
        start = time.monotonic()
        feed.stop()
        assert time.monotonic() - start < 1.0


def camera_log(caplog):
    return [r.getMessage() for r in caplog.records if r.name == 'drones.crazyflie.camera']


def test_the_feed_logs_the_video_going_live_and_dropping(deck, caplog):
    # The server log is where a dropout during a flight can be seen afterwards.
    caplog.set_level(logging.INFO, logger='drones.crazyflie.camera')
    feed = camera.CameraFeed(*deck.getsockname(), retry=0.5).start()
    try:
        conn, _ = deck.accept()
        with conn:
            conn.sendall(image(b'frame', fmt=1))
            wait_for(lambda: feed.status() == 'live')
        wait_for(lambda: feed.status().startswith('no video'))
    finally:
        feed.stop()
    log = camera_log(caplog)
    assert log[:2] == ['AI-deck camera: waiting for frames', 'AI-deck camera: live']
    assert log[2].startswith('AI-deck camera: no video:') and 'closed' in log[2]


def test_a_deck_that_stays_unreachable_is_logged_once_not_every_retry(caplog):
    caplog.set_level(logging.INFO, logger='drones.crazyflie.camera')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    feed = camera.CameraFeed('127.0.0.1', port, retry=0.05).start()
    time.sleep(0.5)                                         # several retries
    feed.stop()
    log = camera_log(caplog)
    assert len(log) == 1 and log[0].startswith('AI-deck camera: no video: Cannot reach')


def test_a_feed_that_never_started_can_still_be_stopped():
    camera.CameraFeed('127.0.0.1', 1).stop()


@needs_cv2
def test_the_feed_encodes_raw_frames_to_jpeg_for_the_browser(deck):
    feed = camera.CameraFeed(*deck.getsockname(), retry=0.05).start()
    try:
        conn, _ = deck.accept()
        with conn:
            conn.sendall(image(bytes(range(48)), width=8, height=6))
            wait_for(lambda: feed.latest()[0] == 1)
            assert feed.latest()[1][:2] == b'\xff\xd8'      # the JPEG start-of-image marker
    finally:
        feed.stop()


@needs_cv2
def test_a_window_opencv_has_torn_down_counts_as_closed(monkeypatch):
    # OpenCV's Qt build raises "NULL guiReceiver" once its last window is gone.
    import cv2

    def gone(*_args):
        raise cv2.error('NULL guiReceiver (please create a window)')
    monkeypatch.setattr(cv2, 'getWindowProperty', gone)
    assert camera._window_closed('AI-deck camera')


@needs_cv2
def test_raw_frames_are_demosaiced_to_colour_or_kept_grey():
    frame = camera.Frame(8, 6, 1, camera.FORMAT_RAW, bytes(range(48)))
    assert camera.decode(frame).shape == (6, 8, 3)
    assert camera.decode(frame, mono=True).shape == (6, 8)


@needs_cv2
def test_a_raw_frame_of_the_wrong_size_is_rejected():
    with pytest.raises(ValueError, match='8x6'):
        camera.decode(camera.Frame(8, 6, 1, camera.FORMAT_RAW, bytes(40)))


@needs_cv2
def test_jpeg_frames_decode_and_garbage_does_not():
    import cv2

    _, jpeg = cv2.imencode('.jpg', np.full((6, 8, 3), 128, np.uint8))
    assert camera.decode(camera.Frame(8, 6, 1, 1, jpeg.tobytes())).shape == (6, 8, 3)
    with pytest.raises(ValueError, match='did not decode'):
        camera.decode(camera.Frame(8, 6, 1, 1, b'not a jpeg'))
