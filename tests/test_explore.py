"""drones-explore-scene: the keyboard pilot, and a hidden window flying a synthetic room.

The pilot is tested against an analytic room -- a floor at z = 0 and a wall at x = WALL -- so it
needs neither a window nor a simulator. The window test needs a desktop session and skips without.
"""
import math
import os

import numpy as np
import pytest

pytest.importorskip('crazyflow')

from drones.sim.explore import (LEASH, MIN_HEIGHT, RADIUS, START_HEIGHT, TURN, Intent, Pilot,
                                list_scenes)

WALL = 1.0
DT = 0.02


def room_distance(origin, direction):
    """Distance along a unit axis ray to the floor (z = 0) or the wall (x = WALL)."""
    hits = []
    if direction[0] > 0:
        hits.append((WALL - origin[0]) / direction[0])
    if direction[2] < 0:
        hits.append(-origin[2] / direction[2])
    hits = [h for h in hits if h >= 0]
    return min(hits) if hits else math.inf


def fly(pilot, intent, seconds, follow=True):
    """Run the pilot; the drone follows the setpoint exactly unless follow=False."""
    drone = pilot.setpoint.copy()
    for _ in range(round(seconds / DT)):
        command = pilot.update(intent, DT, drone)
        if follow:
            drone = command[:3].copy()
    return command


def test_keys_move_in_the_body_frame():
    pilot = Pilot(room_distance, start=(0.0, 0.0, 1.0))
    fly(pilot, Intent(left=1.0), 0.5)             # A: +y at yaw 0
    assert pilot.setpoint[1] > 0.3 and abs(pilot.setpoint[0]) < 1e-9
    fly(pilot, Intent(turn=1.0), 1.0)             # Left arrow: yaw grows, turning left
    assert pilot.yaw == pytest.approx(TURN * 1.0)
    y = pilot.setpoint[1]
    fly(pilot, Intent(forward=1.0), 0.5)          # now facing +y
    assert pilot.setpoint[1] > y + 0.3


def test_command_is_crazyflow_state_layout():
    pilot = Pilot(room_distance, start=(0.0, 0.0, 1.0), yaw=0.5)
    command = pilot.update(Intent(forward=1.0), DT, pilot.setpoint)
    assert command.shape == (13,)
    np.testing.assert_allclose(command[0:3], pilot.setpoint)
    assert command[3] > 0 and command[9] == pytest.approx(0.5)   # velocity, then yaw at [9]


def test_a_wall_stops_the_setpoint_and_the_other_axes_slide():
    pilot = Pilot(room_distance, start=(0.0, 0.0, 1.0))
    fly(pilot, Intent(forward=1.0, left=1.0), 4.0)
    assert pilot.setpoint[0] <= WALL - RADIUS
    assert pilot.setpoint[0] > WALL - RADIUS - 0.05
    assert pilot.setpoint[1] > 2.0                # still sliding along the wall


def test_descending_stops_above_the_floor():
    pilot = Pilot(room_distance, start=(0.0, 0.0, 1.0))
    fly(pilot, Intent(up=-1.0, fast=2.0), 5.0)
    assert pilot.setpoint[2] >= max(MIN_HEIGHT, RADIUS) - 1e-9


def test_the_leash_limits_the_keys_but_keeps_the_take_off_height():
    pilot = Pilot(room_distance)
    grounded = np.zeros(3)
    # The drone has not left the floor yet: the take-off setpoint must stay where it is ...
    command = pilot.update(Intent(), DT, grounded)
    assert command[2] == START_HEIGHT
    # ... while the keys cannot push the setpoint more than LEASH ahead of a drone that lags.
    for _ in range(100):
        pilot.update(Intent(left=1.0), DT, grounded)
    assert pilot.setpoint[1] <= LEASH + 1e-9


def test_list_scenes_puts_the_chosen_scene_first(tmp_path, monkeypatch):
    import drones.sim.explore as explore

    for name in ('A', 'B', 'C'):
        (tmp_path / f'{name}.glb').write_bytes(b'')
    monkeypatch.setattr(explore, 'SCENES_DIR', tmp_path)
    assert [p.stem for p in list_scenes(None)] == ['A', 'B', 'C']
    assert [p.stem for p in list_scenes(tmp_path / 'B.glb')] == ['B', 'A', 'C']


@pytest.mark.skipif(not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')),
                    reason='needs a desktop session for a GLFW window')
def test_a_hidden_window_flies_a_scene_and_starts_questions_at_their_pose(tmp_path):
    # One window for everything: a second GLFW window in the same process, after the first was
    # terminated, reads back black, which drones-explore-scene itself never does.
    import time

    import mujoco

    from drones.sim.eqa import Question
    from drones.sim.explore import Explorer, Stop
    from test_scenes import room_mesh, room_uv, write_glb

    vertices, faces = room_mesh()
    image = np.full((64, 64, 3), 180, np.uint8)
    path = tmp_path / 'Room.glb'
    write_glb(path, vertices, faces.reshape(-1), room_uv(vertices), image)
    open_floor = Question('a-eqa', 1, 'Room', 'What is on the bed?', 'A pillow', 'object')
    posed = Question('hm-eqa', 7, 'Room', 'Is the lamp on?', 'B) Yes', 'existence',
                     ('A) No', 'B) Yes'), start=np.array([0.0, 1.0, 0.3]), yaw=0.5)
    # IndoorUAV's start is the drone in the air, 1.3 m over the room's floor (file z = 0.3).
    flying = Question('indoor-uav', 9, 'Room', 'Fly to the pillar. ' * 60, 'In detail ' * 200,
                      'traj_1, train, easy', start=np.array([0.0, -1.0, 1.6]), yaw=0.0,
                      path=np.array([[0.0, -1.0, 1.6], [2.0, -1.0, 1.6]]),
                      goal=np.array([2.0, -1.0, 1.6]), reveal='detailed instruction')

    explorer = Explorer([Stop(path, [open_floor, posed, flying])], 320, 240, visible=False)
    try:
        # Without a start pose the scan stays where load() put it: the open floor at the origin.
        np.testing.assert_allclose(explorer.anchor, explorer.scene.origin)
        deadline = time.perf_counter() + 20
        while explorer.sim_time < 3.0 and time.perf_counter() < deadline:
            explorer.advance()
            explorer.draw()
        assert explorer.sim_time >= 3.0
        assert abs(explorer.drone()[0][2] - START_HEIGHT) < 0.2
        width, height = explorer.glfw.get_framebuffer_size(explorer.window)
        rgb = np.zeros((height, width, 3), np.uint8)
        mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, width, height), explorer.context)
        assert rgb.std() > 5   # a picture, not a blank buffer

        explorer._pressed.append(explorer.glfw.KEY_RIGHT_BRACKET)   # next question
        explorer.handle_presses()
        assert explorer.question is posed and not explorer.show_answer
        assert explorer.pilot.yaw == pytest.approx(0.5)
        np.testing.assert_allclose(explorer.drone()[0], 0.0, atol=1e-6)   # restarted
        # The scan moved so the question's start is the world origin: its floor point is below
        # the drone, and the rays see the pillar face (file x = 3) 3 m ahead along +x.
        mujoco.mj_kinematics(explorer.model, explorer.data)
        ahead = explorer.pilot.distance(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
        assert ahead == pytest.approx(3.0, abs=0.01)
        assert 'Is the lamp on?' in explorer._question_text(width)
        assert 'B) Yes' in explorer._question_text(width)
        assert 'Answer' not in explorer._question_text(width)
        explorer._pressed.append(explorer.glfw.KEY_SPACE)
        explorer.handle_presses()
        assert 'Answer: B) Yes' in explorer._question_text(width)
        explorer.draw()

        explorer._pressed.append(explorer.glfw.KEY_RIGHT_BRACKET)
        explorer.handle_presses()
        assert explorer.question is flying
        # Anchored on the floor under the start, taking off to the start's own height.
        np.testing.assert_allclose(explorer.anchor, [0.0, -1.0, 0.3], atol=0.02)
        assert explorer.takeoff == pytest.approx(1.3, abs=0.02)
        assert explorer.pilot.setpoint[2] == pytest.approx(1.3, abs=0.02)
        assert 'Space: show the detailed instruction' in explorer._question_text(width)
        # Longer than mjr_overlay's 500 characters and than the window: drawn, and scrollable.
        explorer._pressed += [explorer.glfw.KEY_SPACE, explorer.glfw.KEY_PAGE_DOWN]
        explorer.handle_presses()
        explorer.draw()
        assert explorer.scroll > 0
    finally:
        explorer.glfw.terminate()


def test_ascii_text_keeps_what_mujoco_fonts_can_draw():
    from drones.sim.explore import ascii_text

    assert ascii_text('the hallway’s “end” — café…') == 'the hallway\'s "end" - cafe...'
