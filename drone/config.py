"""Configuration, loaded from the `.env` file next to the project root."""
import os

from dotenv import load_dotenv

load_dotenv()


def _f(name, default):
    return float(os.getenv(name, default))


# A literal URI (radio://0/80/2M/E7E7E7E7E7, usb://0), or 'auto' to use
# whatever single interface a scan turns up.
URI = os.getenv('CFLIB_URI', 'auto')

# --- Flight envelope -------------------------------------------------------
TAKEOFF_HEIGHT = _f('TAKEOFF_HEIGHT', '1.0')
TAKEOFF_VELOCITY = _f('TAKEOFF_VELOCITY', '0.3')
MIN_ALTITUDE = _f('MIN_ALTITUDE', '0.2')
MAX_ALTITUDE = _f('MAX_ALTITUDE', '2.0')

# --- Obstacle avoidance ----------------------------------------------------
# A wall closer than this (m) starts pushing the drone away.
AVOID_DISTANCE = _f('AVOID_DISTANCE', '0.9')
# At or below this distance (m) the push is at full strength. Raising it makes
# avoidance firmer earlier; lowering it lets the drone get closer before
# reacting hard.
AVOID_HARD_DISTANCE = _f('AVOID_HARD_DISTANCE', '0.35')
# Fastest the avoidance is allowed to move the drone (m/s).
MAX_AVOID_SPEED = _f('MAX_AVOID_SPEED', '0.6')
# Anything above the drone within this distance (m) counts as a ceiling.
CEILING_DISTANCE = _f('CEILING_DISTANCE', '0.5')

# --- Manual control --------------------------------------------------------
# Forward/back speed at full stick. Deliberately below MAX_AVOID_SPEED so the
# avoidance can still overrule a stick held straight at a wall.
MAX_MANUAL_SPEED = _f('MAX_MANUAL_SPEED', '0.4')
MAX_CLIMB_SPEED = _f('MAX_CLIMB_SPEED', '0.3')
MAX_YAW_RATE = _f('MAX_YAW_RATE', '90.0')

# --- Autonomous mode -------------------------------------------------------
MAX_FLIGHT_TIME = _f('MAX_FLIGHT_TIME', '60.0')

# --- Web server ------------------------------------------------------------
WEB_HOST = os.getenv('WEB_HOST', '0.0.0.0')
WEB_PORT = int(os.getenv('WEB_PORT', '8000'))
# If non-empty, clients must supply ?token=... to load the page or open the
# WebSocket. Anyone on your LAN can fly the drone otherwise.
WEB_TOKEN = os.getenv('WEB_TOKEN', '')

# Seconds without a client message before the drone stops moving / lands.
STICK_TIMEOUT = _f('STICK_TIMEOUT', '0.7')
LINK_TIMEOUT = _f('LINK_TIMEOUT', '3.0')
