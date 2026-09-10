'use strict';

// Control frames per second. Also the watchdog heartbeat, so it must stay
// comfortably above 1 / STICK_TIMEOUT on the server.
const SEND_HZ = 20;
const FLYING_STATES = ['taking_off', 'flying', 'landing'];
// Ranger distance (m) below which a readout is highlighted.
const NEAR = 0.6;

const $ = (id) => document.getElementById(id);
const token = new URLSearchParams(location.search).get('token') || '';

let socket = null;
let retries = 0;
let latest = { state: 'disconnected', ranges: {} };

// Manual input: forward/back and turn as -1..1, plus an absolute target
// height in metres. Height stays null until the drone tells us where it is,
// so a stale handle position can never command a jump on take-off.
const control = { forward: 0, yaw: 0, altitude: null };
let limits = { min: 0.2, max: 2.0 };
let draggingHeight = false;

/* ---------------------------------------------------------------- socket */

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const query = token ? `?token=${encodeURIComponent(token)}` : '';
  socket = new WebSocket(`${proto}://${location.host}/ws${query}`);

  socket.onopen = () => {
    retries = 0;
    setBar('warn', 'link up', 'Connected to the server');
  };
  socket.onmessage = (event) => render(JSON.parse(event.data));
  socket.onclose = (event) => {
    // The close code is the only clue to why a phone dropped the link, so
    // put it on screen rather than only in the server log.
    const detail = `code ${event.code}${event.reason ? ` ${event.reason}` : ''}`;
    retries += 1;
    const delay = Math.min(1000 * retries, 5000);
    setBar('bad', 'link lost',
           `Lost server link (${detail}), retry ${retries} in ${delay / 1000}s`);
    setTimeout(connect, delay);
  };
}

function send(payload) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
  }
}

setInterval(() => send({ type: 'control', ...control }), 1000 / SEND_HZ);

/* ------------------------------------------------------------------- ui  */

const fmt = (v) => (v === null || v === undefined ? '—' : `${v.toFixed(2)} m`);

function setBar(dotClass, stateText, messageText) {
  $('dot').className = `dot ${dotClass}`;
  $('state').textContent = stateText;
  if (messageText !== undefined) $('message').textContent = messageText;
}

function render(data) {
  latest = data;
  if (data.limits) limits = data.limits;

  const flying = FLYING_STATES.includes(data.state);
  const dot = data.state === 'error' ? 'bad'
    : flying ? 'ok'
    : data.state === 'disconnected' ? '' : 'warn';
  setBar(dot, data.state.replace('_', ' '), data.message);

  const ranges = data.ranges || {};
  let anyNear = false;
  for (const side of ['front', 'back', 'left', 'right', 'up']) {
    const element = $(`r-${side}`);
    const value = ranges[side];
    element.textContent = fmt(value);
    const near = value !== null && value !== undefined && value < NEAR;
    element.classList.toggle('near', near);
    anyNear = anyNear || near;
  }
  $('craft').classList.toggle('warn', anyNear);
  renderHeight(data, ranges);

  $('battery').textContent = data.battery ? `${data.battery.toFixed(2)} V` : '—';
  // 3.2 V is roughly where a Crazyflie should already be on the ground.
  $('battery').classList.toggle('low', data.battery && data.battery < 3.2);

  $('btn-auto').classList.toggle('on', data.auto);
  $('btn-avoid').classList.toggle('on', data.avoid);
  $('btn-avoid').textContent = data.avoid ? 'Avoidance on' : 'Avoidance OFF';

  const connected = data.state !== 'disconnected' && data.state !== 'connecting';
  $('btn-connect').textContent = connected ? 'Disconnect' : 'Connect';
  $('btn-takeoff').disabled = data.state !== 'idle';
  $('btn-land').disabled = !flying;
  $('btn-recover').disabled = !connected || flying;

  // Manual controls do nothing in autonomous mode or on the ground; show
  // that rather than silently swallowing the input.
  const manual = flying && !data.auto;
  for (const el of document.querySelectorAll('.pad, .slider')) {
    el.classList.toggle('disabled', !manual);
  }
  if (!manual) releaseFly();
}

function pctFor(metres) {
  const span = limits.max - limits.min || 1;
  return Math.max(0, Math.min(1, (metres - limits.min) / span)) * 100;
}

function renderHeight(data, ranges) {
  // Adopt the drone's own target while not dragging, so the handle follows
  // take-off and landing instead of fighting them.
  if (!draggingHeight) control.altitude = data.desired_altitude;

  const handle = document.querySelector('#slider-height .handle');
  const fill = document.querySelector('#slider-height .fill');
  const actual = $('height-actual');
  const height = control.altitude;

  if (height === null || height === undefined) {
    handle.style.display = 'none';
    fill.style.height = '0';
    $('height-value').textContent = '—';
  } else {
    const pct = pctFor(height);
    handle.style.display = '';
    handle.style.bottom = `${pct}%`;
    fill.style.height = `${pct}%`;
    $('height-value').textContent = `${height.toFixed(2)} m`;
    $('slider-height').setAttribute('aria-valuenow', height.toFixed(2));
  }

  // Dashed line = measured height, so tracking lag is visible.
  const measured = ranges.down;
  const show = measured !== null && measured !== undefined;
  actual.style.display = show ? '' : 'none';
  if (show) actual.style.bottom = `${pctFor(measured)}%`;
}

/* -------------------------------------------------------------- controls */

function pointerControl(element, { onMove, onRelease }) {
  let pointerId = null;

  const update = (event) => onMove(event, element.getBoundingClientRect());

  element.addEventListener('pointerdown', (event) => {
    if (element.classList.contains('disabled')) return;
    pointerId = event.pointerId;
    element.setPointerCapture(pointerId);
    element.classList.add('active');
    update(event);
  });
  element.addEventListener('pointermove', (event) => {
    if (event.pointerId === pointerId) update(event);
  });
  for (const kind of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    element.addEventListener(kind, (event) => {
      if (event.pointerId === pointerId) {
        pointerId = null;
        element.classList.remove('active');
        onRelease();
      }
    });
  }
  return () => {
    pointerId = null;
    element.classList.remove('active');
    onRelease();
  };
}

/* Fly pad: up/down = forward/back, left/right = turn. Springs to centre. */
const flyPad = $('pad-fly');
const flyKnob = flyPad.querySelector('.knob');

const releaseFly = pointerControl(flyPad, {
  onMove: (event, box) => {
    const halfW = box.width / 2;
    const halfH = box.height / 2;
    let dx = (event.clientX - (box.left + halfW)) / halfW;
    let dy = (event.clientY - (box.top + halfH)) / halfH;
    dx = Math.max(-1, Math.min(1, dx));
    dy = Math.max(-1, Math.min(1, dy));
    // Keep the knob inside the pad rather than centred on the fingertip.
    flyKnob.style.transform =
      `translate(${-50 + dx * 82}%, ${-50 + dy * 82}%)`;
    // Screen up is forward; screen right is a right turn, which is -yaw.
    control.forward = -dy;
    control.yaw = -dx;
  },
  onRelease: () => {
    flyKnob.style.transform = 'translate(-50%, -50%)';
    control.forward = 0;
    control.yaw = 0;
  },
});

/* Height slider: absolute target in metres, stays where you leave it. */
pointerControl($('slider-height'), {
  onMove: (event, box) => {
    draggingHeight = true;
    const fraction = 1 - (event.clientY - box.top) / box.height;
    const clamped = Math.max(0, Math.min(1, fraction));
    control.altitude = limits.min + clamped * (limits.max - limits.min);
    renderHeight({ desired_altitude: control.altitude }, latest.ranges || {});
  },
  onRelease: () => { draggingHeight = false; },
});

function centreSticks() {
  releaseFly();
  draggingHeight = false;
}

/* --------------------------------------------------------------- buttons */

const button = (id, handler) => $(id).addEventListener('click', handler);

button('btn-connect', () => {
  const connected = latest.state !== 'disconnected' && latest.state !== 'connecting';
  send({ type: connected ? 'disconnect' : 'connect' });
});
button('btn-takeoff', () => {
  centreSticks();
  send({ type: 'takeoff', value: latest.auto });
});
button('btn-land', () => { centreSticks(); send({ type: 'land' }); });
button('btn-recover', () => send({ type: 'recover' }));
button('btn-auto', () => send({ type: 'auto', value: !latest.auto }));
button('btn-avoid', () => send({ type: 'avoid', value: !latest.avoid }));
button('btn-estop', () => { centreSticks(); send({ type: 'estop' }); });

// Backgrounding the page freezes our timers, so drop the sticks on the way
// out rather than leaving a stale input for the server to act on.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) centreSticks();
});
window.addEventListener('blur', centreSticks);

connect();
