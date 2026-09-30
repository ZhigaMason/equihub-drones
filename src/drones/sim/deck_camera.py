"""What the AI-deck sees in simulation, through the lens calibrated on the real deck.

`sensors.render_camera` raycasts a flat-shaded 32x24 image inside the jitted env, cheap enough to
train on. This is the other camera: an offscreen MuJoCo render of one world, with the lighting,
textures and any scanned scene (`scenes.attach`) that Sim.render draws, projected through the
intrinsics measured on the deck (drones.sim.lens). A frame here has the size, principal point and
distortion of a frame from `drones-fpv --record`.

The camera's own drone is left out of its view: from where the deck sits, the arms and props of
CrazyFlow's model fill the lower corners, and a real deck's frames show none of them.
"""
import mujoco
import numpy as np

from drones.sim.lens import DECK_MOUNT, SUPERSAMPLE, LensCamera, quat_to_matrix


class DeckCamera:
    """Offscreen frames from a camera fixed to the drone of one world of a HoverEnv or SquareEnv.

    Build it after anything that replaces env.sim.mj_model (`scenes.attach`). Use as a context
    manager, or call close().
    """

    def __init__(self, env, intrinsics, world=0, mount=DECK_MOUNT, supersample=SUPERSAMPLE):
        self.env, self.world, self.mount = env, world, mount
        self.width, self.height = intrinsics.width, intrinsics.height
        self.model = model = env.sim.mj_model
        self.data = mujoco.MjData(model)
        mocap = int(np.asarray(env.sim.data.core.drone_mocap_ids)[0])
        body = int(np.flatnonzero(model.body_mocapid == mocap)[0])
        own = np.flatnonzero(model.body_rootid[model.geom_bodyid] == body)
        self.lens = LensCamera(model, intrinsics, supersample, hidden_geoms=own)

    def reset(self):
        """Nothing to forget between episodes; here to stand in for a TrajectoryRenderer."""

    def frame(self, state, trail=(), hud=(), path=(), target=None):
        """An RGB image (height, width, 3) of what the deck sees in `state`, an EnvState.

        Takes TrajectoryRenderer.frame's arguments so the render CLIs can use either, but draws
        no markers or text: the image holds only what a real deck would see.
        """
        sim, w = self.env.sim, self.world
        # As TrajectoryRenderer does: the env keeps its state outside the Sim, and HoverEnv carries
        # its own `mjx` with the walls it moved this episode.
        sim.data = state.sim
        if hasattr(state, 'mjx'):
            sim.mjx_data = state.mjx
        if not sim.data.core.mjx_synced:
            from crazyflow.sim.sim import sync_sim2mjx
            sim.data, sim.mjx_data = sync_sim2mjx(sim.data, sim.mjx_data, sim.mjx_model)
        d = self.data
        d.qpos[:] = sim.mjx_data.qpos[w]
        d.mocap_pos[:] = sim.mjx_data.mocap_pos[w]
        d.mocap_quat[:] = sim.mjx_data.mocap_quat[w]
        # Not mj_forward: it raises on contacts between the drones welded to the world.
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        mujoco.mj_camlight(self.model, d)
        states = state.sim.states
        return self.render(d, np.asarray(states.pos[w, 0], float),
                           np.asarray(states.quat[w, 0], float))

    def render(self, data, pos, quat):
        """What the deck sees from a drone at `pos` with attitude `quat` (scalar-last), in `data`,
        an MjData of this camera's model already put through mj_kinematics."""
        return self.lens.render(data, *self.mount.camera_pose(pos, quat_to_matrix(quat)))

    def close(self):
        self.lens.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
