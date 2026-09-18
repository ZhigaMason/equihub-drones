"""The phone page over a real HTTP server and a real WebSocket handshake.

Deliberately not Starlette's TestClient: it fakes the WebSocket transport,
which once let a missing WebSocket library pass every test.
"""
import asyncio
import json
import socket
import threading
import time

import httpx2 as httpx
import pytest
import uvicorn
import websockets

from drones import config
from drones.teleop.web import server as web


@pytest.fixture(scope='module')
def host():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    srv = uvicorn.Server(web.uvicorn_config('127.0.0.1', port))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started:
        assert time.time() < deadline, 'server did not start'
        time.sleep(0.05)
    yield f'127.0.0.1:{port}'
    srv.should_exit = True
    thread.join(timeout=10)


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=10))


@pytest.mark.parametrize('path', ['/', '/static/app.js', '/static/style.css'])
def test_ui_is_served_uncached(host, path):
    response = httpx.get(f'http://{host}{path}')
    assert response.status_code == 200
    assert 'no-store' in response.headers['cache-control']


def test_conditional_request_is_never_answered_304(host):
    response = httpx.get(f'http://{host}/static/app.js',
                         headers={'If-None-Match': '*'})
    assert response.status_code == 200


def test_page_has_the_current_controls(host):
    page = httpx.get(f'http://{host}/').text
    assert 'pad-fly' in page and 'slider-height' in page and 'fpv-img' in page


def test_token_gates_page_and_socket(host, monkeypatch):
    monkeypatch.setattr(config, 'WEB_TOKEN', 'sekret')
    assert httpx.get(f'http://{host}/').status_code == 403
    assert httpx.get(f'http://{host}/?token=sekret').status_code == 200

    async def without_token():
        async with websockets.connect(f'ws://{host}/ws') as ws:
            await ws.recv()

    with pytest.raises((websockets.exceptions.InvalidStatus,
                        websockets.exceptions.ConnectionClosed)):
        run(without_token())


def test_telemetry_streams_at_about_ten_hertz(host):
    async def session():
        async with websockets.connect(f'ws://{host}/ws') as ws:
            first = json.loads(await ws.recv())
            frames, start = 0, time.time()
            while time.time() - start < 1.0:
                await ws.recv()
                frames += 1
            return first, frames

    first, frames = run(session())
    assert {'state', 'ranges', 'limits', 'desired_altitude'} <= first.keys()
    assert frames >= 7


def test_control_frames_reach_the_controller(host):
    async def session():
        async with websockets.connect(f'ws://{host}/ws') as ws:
            await ws.recv()
            await ws.send(json.dumps({'type': 'control', 'forward': 0.8,
                                      'yaw': -0.4, 'altitude': 1.3}))
            await asyncio.sleep(0.3)

    run(session())
    assert web.controller._forward == pytest.approx(0.8)
    assert web.controller._yaw == pytest.approx(-0.4)
    assert web.controller._desired_altitude == pytest.approx(1.3)


def test_malformed_frame_does_not_drop_the_link(host):
    async def session():
        async with websockets.connect(f'ws://{host}/ws') as ws:
            await ws.recv()
            await ws.send('not json at all')
            await ws.send(json.dumps({'type': 'control', 'forward': -0.25,
                                      'yaw': 0}))
            await asyncio.sleep(0.3)
            return json.loads(await ws.recv())

    snapshot = run(session())
    assert 'state' in snapshot
    assert web.controller._forward == pytest.approx(-0.25)


class FakeFeed:
    """Stands in for CameraFeed under drones-fpv: one fixed JPEG, a fixed status, no deck."""

    def __init__(self, status='live'):
        self.jpeg = b'\xff\xd8 not really a picture \xff\xd9'
        self._status = status

    def start(self):
        return self

    def stop(self):
        pass

    def latest(self):
        return 1, self.jpeg

    def status(self):
        return self._status


@pytest.fixture
def feed(monkeypatch):
    fake = FakeFeed()
    monkeypatch.setattr(web.app.state, 'feed', fake)
    return fake


def first_telemetry(host):
    async def session():
        async with websockets.connect(f'ws://{host}/ws') as ws:
            return json.loads(await ws.recv())
    return run(session())


def first_video_part(url, expect):
    with httpx.stream('GET', url, timeout=5) as response:
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('multipart/x-mixed-replace')
        received = b''
        for piece in response.iter_bytes():
            received += piece
            if expect in received:
                return received
    raise AssertionError(f'the stream ended without the frame; got {received[:80]!r}')


def test_plain_drones_web_has_no_video(host):
    assert httpx.get(f'http://{host}/video').status_code == 404
    assert 'video' not in first_telemetry(host)


def test_video_streams_each_new_frame_as_a_jpeg_part(host, feed):
    part = first_video_part(f'http://{host}/video', feed.jpeg)
    assert b'Content-Type: image/jpeg' in part


def test_token_gates_the_video(host, feed, monkeypatch):
    monkeypatch.setattr(config, 'WEB_TOKEN', 'sekret')
    assert httpx.get(f'http://{host}/video').status_code == 403
    first_video_part(f'http://{host}/video?token=sekret', feed.jpeg)


def test_telemetry_carries_the_video_status_for_the_page(host, monkeypatch):
    monkeypatch.setattr(web.app.state, 'feed', FakeFeed(status='no video: deck unreachable'))
    assert first_telemetry(host)['video'] == 'no video: deck unreachable'


def test_an_open_video_stream_does_not_hold_up_the_landing(feed, monkeypatch):
    # uvicorn runs the lifespan shutdown - controller.stop(), which lands the drone - only after
    # waiting for open connections, and an MJPEG stream never ends by itself. uvicorn 0.52.4 ends
    # it on shutdown even without SHUTDOWN_GRACE; this guards against an upgrade that stops that.
    stopped = threading.Event()
    monkeypatch.setattr(web.controller, 'start', lambda: None)
    monkeypatch.setattr(web.controller, 'stop', stopped.set)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    srv = uvicorn.Server(web.uvicorn_config('127.0.0.1', port))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started:
        assert time.time() < deadline, 'server did not start'
        time.sleep(0.05)

    with httpx.stream('GET', f'http://127.0.0.1:{port}/video', timeout=10) as response:
        next(response.iter_bytes())
        srv.should_exit = True
        assert stopped.wait(timeout=web.SHUTDOWN_GRACE + 3), 'shutdown waited on the stream'
    thread.join(timeout=10)
