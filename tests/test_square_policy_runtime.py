"""The square policy artifact: save, load, and observe exactly as the simulator does."""
import json

import numpy as np
import pytest

from drones.policy.runtime import Policy, PolicySpec, SquareRunner, SquareSpec
from drones.policy.square import OBS_SIZE, heading

SPEC = SquareSpec(control_freq=50, side=1.0, corner_radius=0.15, lap_time=(6.0, 10.0),
                  height=(0.8, 1.2), max_tilt=0.35, max_yaw_rate=1.5, hover_thrust=0.44,
                  thrust_min=0.085, thrust_max=0.8)


def constant_policy(action):
    """A one-layer policy that ignores its input and outputs `action`."""
    return Policy(SPEC, [(np.zeros((OBS_SIZE, 4)), np.asarray(action, float))])


def test_square_artifact_round_trips(tmp_path):
    constant_policy([0.1, 0.2, 0.3, 0.4]).save(tmp_path)
    loaded = Policy.load(tmp_path)
    assert loaded.spec == SPEC
    assert json.loads((tmp_path / 'policy.json').read_text())['task'] == 'square'


def test_version_1_hover_artifacts_still_load(tmp_path):
    spec = dict(sensors=['multiranger', 'optical_flow'], history=3, control_freq=50,
                target_height=[0.5, 1.5], range_max=4.0, flow_gain=0.488, max_tilt=0.35,
                max_yaw_rate=1.5, hover_thrust=0.44, thrust_min=0.085, thrust_max=0.8,
                format_version=1)
    (tmp_path / 'policy.json').write_text(json.dumps(spec))
    size = 3 * 13
    np.savez(tmp_path / 'actor.npz', w0=np.zeros((size, 4)), b0=np.zeros(4))
    loaded = Policy.load(tmp_path)
    assert isinstance(loaded.spec, PolicySpec) and loaded.spec.task == 'hover'


def test_unknown_versions_are_refused(tmp_path):
    constant_policy([0, 0, 0, 0]).save(tmp_path)
    data = json.loads((tmp_path / 'policy.json').read_text())
    (tmp_path / 'policy.json').write_text(json.dumps({**data, 'format_version': 99}))
    with pytest.raises(ValueError, match='format 99'):
        Policy.load(tmp_path)


def test_runner_feeds_back_its_previous_action():
    runner = SquareRunner(constant_policy([0.1, -0.2, 0.3, 2.0]), origin=[0.0, 0.0, 1.0],
                          ref_yaw=0.0, lap_time=8.0)
    state = dict(pos=np.array([0.0, 0.0, 1.0]), vel=np.zeros(3), yaw=0.0,
                 gravity=np.array([0.0, 0.0, -1.0]))
    action = runner.step(0.0, **state)
    np.testing.assert_allclose(action, [0.1, -0.2, 0.3, 1.0], atol=1e-6)   # clipped to 1
    np.testing.assert_allclose(runner.observe(0.02, **state)[-4:], action)


def test_runner_rejects_a_square_too_small_for_its_corners():
    with pytest.raises(ValueError, match='side'):
        SquareRunner(constant_policy([0, 0, 0, 0]), origin=[0, 0, 1], ref_yaw=0.0, lap_time=8.0,
                     side=0.25)


def test_runner_observes_what_the_simulator_does():
    pytest.importorskip('crazyflow')
    import jax

    from drones.missions.fly_policy import gravity_and_tilt
    from drones.sim.square_env import SquareConfig, SquareEnv

    env = SquareEnv(SquareConfig(num_envs=2, pos_noise=0.0, pos_drift=0.0, vel_noise=0.0,
                                 attitude_noise=0.0))
    state, obs = env.reset(jax.random.key(0))
    s = state.sim.states
    runner = SquareRunner(Policy(SquareSpec(**env.policy_spec()),
                                 [(np.zeros((OBS_SIZE, 4)), np.zeros(4))]),
                          origin=np.asarray(state.origin[0]), ref_yaw=float(state.ref_yaw[0]),
                          lap_time=float(state.lap_time[0]),
                          direction=float(state.direction[0]),
                          rotation=float(state.rotation[0]))
    quat = np.asarray(s.quat[0, 0], float)
    gravity, _ = gravity_and_tilt(quat)
    got = runner.observe(float(state.phase[0]), pos=np.asarray(s.pos[0, 0]),
                         vel=np.asarray(s.vel[0, 0]), yaw=float(heading(np, quat)),
                         gravity=gravity)
    np.testing.assert_allclose(got, np.asarray(obs['policy'][0]), atol=1e-4)
