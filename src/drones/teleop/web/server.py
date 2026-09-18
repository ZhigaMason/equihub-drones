"""Self-hosted control page for the Crazyflie, reachable from a phone on the LAN.

Run with:  uv run drones-web
     or:   uv run --extra camera drones-fpv    (the same page, with the AI-deck camera on it)
"""
import argparse
import asyncio
import json
import logging
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from drones import config
from drones.crazyflie.camera import CameraFeed
from drones.crazyflie.controller import DroneController

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-7s %(message)s')
logging.getLogger('cflib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / 'static'


class _NoCacheStatic(StaticFiles):
    """Serve the UI with revalidation forced.

    Phones cache aggressively, and a stale app.js against a new server means
    controls that silently do nothing. This is a LAN tool served off the
    local disk, so the revalidation cost is irrelevant.
    """

    def is_not_modified(self, *args, **kwargs):
        return False

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers['Cache-Control'] = 'no-store, must-revalidate'
        return response


# Telemetry pushes per second.
TELEMETRY_HZ = 10
# Seconds an open connection may hold up shutdown; see uvicorn_config().
SHUTDOWN_GRACE = 1.0
# How often an open /video response looks for a newer frame; the deck sends at most ~30 a second.
VIDEO_POLL = 0.02
BOUNDARY = 'frame'

controller = DroneController()


@asynccontextmanager
async def lifespan(_app):
    controller.start()
    if app.state.feed is not None:
        app.state.feed.start()
    yield
    # The landing first: nothing about video is worth delaying it for.
    controller.stop()
    if app.state.feed is not None:
        app.state.feed.stop()


app = FastAPI(title='Crazyflie control', lifespan=lifespan)
app.mount('/static', _NoCacheStatic(directory=STATIC), name='static')
# A CameraFeed under drones-fpv; None under drones-web, which never touches the AI-deck.
app.state.feed = None


def _authorised(token):
    return not config.WEB_TOKEN or token == config.WEB_TOKEN


@app.get('/')
def index(token: str = Query('')):
    if not _authorised(token):
        return PlainTextResponse('Forbidden: bad or missing token', 403)
    return FileResponse(STATIC / 'index.html',
                        headers={'Cache-Control': 'no-store, must-revalidate'})


@app.get('/video')
def video(token: str = Query('')):
    """The AI-deck camera as MJPEG, which an <img> plays with no script at all."""
    if not _authorised(token):
        return PlainTextResponse('Forbidden: bad or missing token', 403)
    feed = app.state.feed
    if feed is None:
        return PlainTextResponse('No camera here: start the server with drones-fpv', 404)
    return StreamingResponse(_mjpeg(feed),
                             media_type=f'multipart/x-mixed-replace; boundary={BOUNDARY}',
                             headers={'Cache-Control': 'no-store'})


async def _mjpeg(feed):
    seen = 0
    while True:
        count, jpeg = feed.latest()
        if count != seen and jpeg is not None:
            seen = count
            head = (f'--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n'
                    f'Content-Length: {len(jpeg)}\r\n\r\n')
            yield head.encode() + jpeg + b'\r\n'
        await asyncio.sleep(VIDEO_POLL)


@app.websocket('/ws')
async def ws(socket_: WebSocket, token: str = Query('')):
    if not _authorised(token):
        await socket_.close(code=1008)
        return

    await socket_.accept()
    controller.heartbeat()
    logger.info('Client connected from %s', socket_.client)
    pump = asyncio.create_task(_push_telemetry(socket_))
    why = 'closed by client'
    try:
        while True:
            text = await socket_.receive_text()
            try:
                message = json.loads(text)
            except ValueError:
                # One bad frame must not cost us the link, and losing the
                # link mid-flight starts the landing watchdog.
                logger.warning('Ignoring malformed frame: %.80s', text)
                continue
            if isinstance(message, dict):
                _handle(message)
    except WebSocketDisconnect as exc:
        why = f'closed by client (code {exc.code})'
    except Exception as exc:                            # noqa: BLE001
        # Anything else is a bug on our side; do not report it as a normal
        # client disconnect.
        why = f'{type(exc).__name__}: {exc}'
        logger.exception('WebSocket handler failed')
    finally:
        pump.cancel()
        # Do not land here: a phone that briefly drops Wi-Fi reconnects in
        # well under the controller's own link timeout, which is what
        # decides whether the flight is abandoned.
        logger.info('Client disconnected - %s', why)


def _handle(message):
    kind = message.get('type')
    if kind == 'control':
        controller.set_control(message.get('forward', 0),
                               message.get('yaw', 0),
                               message.get('altitude'))
    elif kind == 'ping':
        controller.heartbeat()
    elif kind in ('connect', 'disconnect', 'takeoff', 'land', 'estop',
                  'recover'):
        controller.heartbeat()
        controller.submit(kind, message.get('value'))
    elif kind in ('avoid', 'auto'):
        controller.submit(kind, bool(message.get('value')))


def _telemetry():
    data = controller.snapshot()
    if app.state.feed is not None:
        # Its presence is what tells the page to show the video panel at all.
        data['video'] = app.state.feed.status()
    return data


async def _push_telemetry(socket_):
    period = 1 / TELEMETRY_HZ
    try:
        while True:
            await socket_.send_json(_telemetry())
            await asyncio.sleep(period)
    except asyncio.CancelledError:
        raise
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:                                   # noqa: BLE001
        # A crash here would silently stop telemetry while the socket stayed
        # open, leaving the page frozen on stale numbers.
        logger.exception('Telemetry pump failed')


def _lan_address():
    """Best guess at the address a phone on the same network should use."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Never actually sends anything; just picks the outbound interface.
        probe.connect(('8.8.8.8', 80))
        return probe.getsockname()[0]
    except OSError:
        return '127.0.0.1'
    finally:
        probe.close()


def _check_websocket_support():
    """uvicorn ships no WebSocket implementation of its own, and without one
    it rejects the upgrade with only a log warning - the page loads but every
    control is dead. Fail loudly here instead."""
    try:
        import websockets  # noqa: F401
    except ImportError:
        try:
            import wsproto  # noqa: F401
        except ImportError:
            raise SystemExit(
                'No WebSocket library installed, so the control page would '
                'load but never connect.\nRun:  uv add websockets') from None


def uvicorn_config(host, port):
    """The server's settings, shared by main() and the tests.

    uvicorn runs the lifespan shutdown - controller.stop(), which lands the
    drone - only after waiting for open connections, and with no grace period
    that wait has no limit. An open /video stream never ends by itself.
    Measured on uvicorn 0.52.4 and Starlette 1.6, shutdown ends it anyway,
    landing within 0.2 s with or without the grace period. The grace period is
    the backstop if an upgrade ever changes that.
    """
    return uvicorn.Config(app, host=host, port=port, log_level='warning',
                          timeout_graceful_shutdown=SHUTDOWN_GRACE)


def main():
    _check_websocket_support()
    suffix = f'?token={config.WEB_TOKEN}' if config.WEB_TOKEN else ''
    print(f'\n  Open on your phone:  '
          f'http://{_lan_address()}:{config.WEB_PORT}/{suffix}\n'
          f'  On this computer:    '
          f'http://localhost:{config.WEB_PORT}/{suffix}\n', flush=True)
    if not config.WEB_TOKEN:
        print('  WEB_TOKEN is unset: anyone on this network can fly the '
              'drone.\n', flush=True)
    try:
        uvicorn.Server(uvicorn_config(config.WEB_HOST, config.WEB_PORT)).run()
    except KeyboardInterrupt:
        # uvicorn re-raises the Ctrl-C it handled once shutdown is done;
        # uvicorn.run() swallows it the same way.
        pass


def main_fpv(argv=None):
    """drones-fpv: the same page and controller, with the AI-deck camera on it."""
    parser = argparse.ArgumentParser(
        description='The drones-web control page, with the AI-deck camera feed on it.')
    parser.add_argument('--host', default=config.AIDECK_HOST,
                        help="the AI-deck's address (default %(default)s; AIDECK_HOST)")
    parser.add_argument('--port', type=int, default=config.AIDECK_PORT,
                        help="the deck streamer's TCP port (default %(default)s; AIDECK_PORT)")
    parser.add_argument('--mono', action='store_true',
                        help='the deck has the greyscale Himax: send raw frames without '
                             'demosaicing them into false colour')
    args = parser.parse_args(argv)
    try:
        import cv2  # noqa: F401
    except ImportError:
        raise SystemExit('drones-fpv needs OpenCV to encode raw frames.\n'
                         'Run:  uv run --extra camera drones-fpv') from None

    app.state.feed = CameraFeed(args.host, args.port, mono=args.mono)
    print(f'\n  Camera: AI-deck at {args.host}:{args.port}', flush=True)
    main()


if __name__ == '__main__':
    main()
