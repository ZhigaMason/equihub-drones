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
    srv = uvicorn.Server(uvicorn.Config(web.app, host='127.0.0.1', port=port,
                                        log_level='warning'))
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
    assert 'pad-fly' in page and 'slider-height' in page


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
