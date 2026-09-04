"""Self-hosted control page for the Crazyflie, reachable from a phone on the LAN.

Run with:  uv run python -m drone.server
"""
import asyncio
import json
import logging
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from drone import config
from drone.controller import DroneController

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

controller = DroneController()


@asynccontextmanager
async def lifespan(_app):
    controller.start()
    yield
    controller.stop()


app = FastAPI(title='Crazyflie control', lifespan=lifespan)
app.mount('/static', _NoCacheStatic(directory=STATIC), name='static')


def _authorised(token):
    return not config.WEB_TOKEN or token == config.WEB_TOKEN


@app.get('/')
def index(token: str = Query('')):
    if not _authorised(token):
        return PlainTextResponse('Forbidden: bad or missing token', 403)
    return FileResponse(STATIC / 'index.html',
                        headers={'Cache-Control': 'no-store, must-revalidate'})


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


async def _push_telemetry(socket_):
    period = 1 / TELEMETRY_HZ
    try:
        while True:
            await socket_.send_json(controller.snapshot())
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
                'load but never connect.\nRun:  uv add websockets')


def main():
    _check_websocket_support()
    suffix = f'?token={config.WEB_TOKEN}' if config.WEB_TOKEN else ''
    print(f'\n  Open on your phone:  '
          f'http://{_lan_address()}:{config.WEB_PORT}/{suffix}\n', flush=True)
    if not config.WEB_TOKEN:
        print('  WEB_TOKEN is unset: anyone on this network can fly the '
              'drone.\n', flush=True)
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT,
                log_level='warning')


if __name__ == '__main__':
    main()
