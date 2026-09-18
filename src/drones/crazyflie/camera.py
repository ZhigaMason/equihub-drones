"""Live view of the AI-deck's camera, streamed over Wi-Fi.

The AI-deck never sends images over the Crazyradio. Its GAP8 hands each frame to the deck's ESP32
(NINA) Wi-Fi module, which serves them on a TCP socket, so nothing here uses cflib and the radio
stays free for whatever is flying the drone. The GAP8 must be running Bitcraze's
`wifi-img-streamer` example from aideck-gap8-examples; the wire format below is that example's.

This only reads: it cannot move the aircraft. It does connect to it, though, so like the other
hardware entry points it is for the operator to start.

OpenCV comes from the `camera` extra and is imported only where a frame is decoded or shown, so the
base install can still import this module.
"""
import argparse
import logging
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from drones import config

logger = logging.getLogger(__name__)

# Every CPX packet on the socket starts with its payload length, which counts the routing and
# function bytes that follow it, then those two bytes.
CPX_HEADER = struct.Struct('<HBB')
CPX_ROUTING = 2
# A frame opens with a packet holding only this header; its pixels follow in as many packets as it
# takes to deliver `size` bytes.
IMAGE_HEADER = struct.Struct('<BHHBBI')   # magic, width, height, depth, format, size
IMAGE_MAGIC = 0xBC
FORMAT_RAW = 0          # the sensor's raw pixels; any other format is a JPEG

CONNECT_TIMEOUT = 5.0   # s to open the socket
# s without a byte before the stream counts as dead. A stall is almost always the deck's ESP out of
# memory and deadlocked, not this code: a client whose Wi-Fi power saving is on makes its access
# point buffer frames until the ~48 KB heap is gone. Reconnecting rarely revives it; replugging the
# drone does. Turning power saving off for the drone's network prevents it (README, "The deck on
# this drone").
STALL_TIMEOUT = 5.0
RETRY_PERIOD = 2.0      # s between a CameraFeed's attempts to reach the deck again


@dataclass(frozen=True)
class Frame:
    width: int
    height: int
    depth: int
    format: int
    data: bytes


def connect(host, port, timeout=CONNECT_TIMEOUT):
    """A socket to the deck's streamer, with an error that says what to check when it fails."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot reach the AI-deck at {host}:{port} ({exc}). Join the deck's Wi-Fi network, "
            f'or set AIDECK_HOST in .env to the address it took on yours.') from exc
    sock.settimeout(STALL_TIMEOUT)
    return sock


def _read_exactly(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError('The AI-deck closed the connection')
        data += chunk
    return bytes(data)


def read_packet(sock):
    """The payload of the next CPX packet, without its routing and function bytes."""
    length, _routing, _function = CPX_HEADER.unpack(_read_exactly(sock, CPX_HEADER.size))
    return _read_exactly(sock, max(0, length - CPX_ROUTING))


def read_frame(sock):
    """The next whole frame.

    Packets before an image header are skipped, so joining the stream mid-frame costs one frame
    rather than showing a garbled one.
    """
    while True:
        payload = read_packet(sock)
        if len(payload) == IMAGE_HEADER.size and payload[0] == IMAGE_MAGIC:
            break
    _magic, width, height, depth, fmt, size = IMAGE_HEADER.unpack(payload)
    data = bytearray()
    while len(data) < size:
        data += read_packet(sock)
    return Frame(width, height, depth, fmt, bytes(data[:size]))


def decode(frame, mono=False):
    """A frame as an image OpenCV can show: BGR, or single-channel with `mono`."""
    import cv2

    pixels = np.frombuffer(frame.data, dtype=np.uint8)
    if frame.format != FORMAT_RAW:
        image = cv2.imdecode(pixels, cv2.IMREAD_GRAYSCALE if mono else cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'A {len(frame.data)}-byte JPEG frame did not decode')
        return image
    if pixels.size != frame.width * frame.height:
        raise ValueError(
            f'A raw {frame.width}x{frame.height} frame arrived with {pixels.size} bytes')
    mosaic = pixels.reshape(frame.height, frame.width)
    # The colour Himax sits behind a Bayer filter, and BG is the pattern Bitcraze's own viewer
    # (opencv-viewer.py) demosaics with. On the greyscale sensor the mosaic already is the image.
    return mosaic if mono else cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2BGR)


def to_jpeg(frame, mono=False):
    """The frame as JPEG bytes, for a browser. A JPEG frame passes through untouched."""
    if frame.format != FORMAT_RAW:
        return frame.data
    import cv2

    ok, encoded = cv2.imencode('.jpg', decode(frame, mono))
    if not ok:
        raise ValueError(f'A raw {frame.width}x{frame.height} frame did not encode to JPEG')
    return encoded.tobytes()


class FrameStream:
    """Reads frames on a thread and keeps only the newest.

    Reading as fast as the deck sends is what keeps the view live: a window that drew slower than
    the stream would otherwise leave frames queued in the socket and fall ever further behind.
    """

    def __init__(self, sock):
        self._sock = sock
        self._lock = threading.Lock()
        self._count = 0
        self._frame = None
        self._error = None
        self._closing = False
        self._thread = threading.Thread(target=self._run, name='aideck-stream', daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        try:
            while True:
                frame = read_frame(self._sock)
                with self._lock:
                    self._count, self._frame = self._count + 1, frame
        except Exception as exc:    # handed to the caller by latest()
            if not self._closing:
                with self._lock:
                    self._error = exc

    def latest(self):
        """(frames received so far, the newest one or None); raises whatever stopped the reader."""
        with self._lock:
            if self._error is not None:
                raise self._error
            return self._count, self._frame

    def close(self):
        self._closing = True
        try:
            self._sock.shutdown(socket.SHUT_RDWR)   # wakes the reader out of recv()
        except OSError:
            pass
        self._sock.close()
        self._thread.join(timeout=1.0)


class CameraFeed:
    """The deck's stream kept alive for a server: it reconnects on its own and holds the newest
    frame as JPEG.

    It runs on its own thread and never raises into its caller, so a deck that is off, out of
    range or on another network costs the page its picture and nothing else. What went wrong is
    in `status()` instead.
    """

    def __init__(self, host, port, mono=False, retry=RETRY_PERIOD):
        self.host, self.port, self.mono, self.retry = host, port, mono, retry
        self._lock = threading.Lock()
        self._count = 0
        self._jpeg = None
        self._status = 'connecting'
        self._logged = None         # the last status logged; the feed thread's alone
        self._sock = None
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, name='aideck-feed', daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stopping.set()
        with self._lock:
            sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)     # wakes the reader out of recv()
            except OSError:
                pass
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def latest(self):
        """(frames received so far, the newest one as JPEG bytes or None)."""
        with self._lock:
            return self._count, self._jpeg

    def status(self):
        """'connecting', 'waiting for frames', 'live', or 'no video: <why>'."""
        with self._lock:
            return self._status

    def _set(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, '_' + key, value)

    def _report(self, status):
        self._set(status=status)
        self._log(status)

    def _log(self, status):
        """Log a status that says something new.

        'connecting' is passed over, and a failure that repeats on every retry is logged once, so
        a deck that stays switched off costs the log one line, while a feed that keeps dropping
        mid-flight shows every drop.
        """
        if status != 'connecting' and status != self._logged:
            self._logged = status
            logger.info('AI-deck camera: %s', status)

    def _run(self):
        while not self._stopping.is_set():
            self._report('connecting')
            try:
                with connect(self.host, self.port) as sock:
                    # stop() reads _sock after setting _stopping, so checking _stopping after
                    # publishing the socket leaves no window where neither side sees the other.
                    self._set(sock=sock)
                    self._report('waiting for frames')
                    while not self._stopping.is_set():
                        jpeg = to_jpeg(read_frame(sock), self.mono)
                        # Status with the frame, so whoever sees the frame also sees 'live'.
                        with self._lock:
                            self._count, self._jpeg = self._count + 1, jpeg
                            self._status = 'live'
                        self._log('live')
            except Exception as exc:    # the page shows it; the server carries on
                if not self._stopping.is_set():
                    self._report(f'no video: {exc or type(exc).__name__}')
            finally:
                self._set(sock=None)
            self._stopping.wait(self.retry)


def _window_closed(window):
    import cv2

    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        # OpenCV's Qt build tears its GUI down with the last window, then raises here
        # ("NULL guiReceiver") instead of reporting the window as not visible.
        return True


def _annotate(image, text):
    import cv2

    for colour, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(image, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thickness,
                    cv2.LINE_AA)
    return image


def view(host, port, scale=2, mono=False):
    """Show the stream in a window until Q, Esc, the window's close button, or a dead stream."""
    import cv2

    window = f'AI-deck camera ({host})'
    stream = FrameStream(connect(host, port)).start()
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    shown, fps, mark = 0, 0.0, (time.monotonic(), 0)
    try:
        while True:
            count, frame = stream.latest()
            now = time.monotonic()
            if now - mark[0] >= 1.0:
                fps, mark = (count - mark[1]) / (now - mark[0]), (now, count)
            if count != shown:
                image = cv2.resize(decode(frame, mono), None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_NEAREST)
                kind = 'raw' if frame.format == FORMAT_RAW else 'jpeg'
                cv2.imshow(window, _annotate(
                    image, f'{fps:4.1f} fps  {frame.width}x{frame.height} {kind}'))
                shown = count
            if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
                return
            # Before the first imshow some backends report the window as not yet visible.
            if shown and _window_closed(window):
                return
    finally:
        stream.close()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Show the AI-deck camera stream in a window. Q or Esc quits.')
    parser.add_argument('--host', default=config.AIDECK_HOST,
                        help="the deck's address (default %(default)s; AIDECK_HOST in .env)")
    parser.add_argument('--port', type=int, default=config.AIDECK_PORT,
                        help='the streamer\'s TCP port (default %(default)s; AIDECK_PORT)')
    parser.add_argument('--scale', type=int, default=2,
                        help='enlarge the image by this whole factor (default %(default)s)')
    parser.add_argument('--mono', action='store_true',
                        help='the deck has the greyscale Himax: show raw frames without '
                             'demosaicing them into false colour')
    args = parser.parse_args(argv)
    if args.scale < 1:
        parser.error('--scale must be at least 1')

    try:
        import cv2  # noqa: F401
    except ImportError:
        print('Error: the viewer needs OpenCV: uv run --extra camera drones-camera',
              file=sys.stderr)
        return 1

    print(f'Connecting to the AI-deck at {args.host}:{args.port}', flush=True)
    try:
        view(args.host, args.port, args.scale, args.mono)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
