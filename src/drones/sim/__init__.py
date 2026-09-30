"""CrazyFlow simulation of the Crazyflie 2.1 Brushless: scene, sensors and training tasks.

Needs the optional ``sim`` extra: ``uv sync --extra sim``.
"""
import os
import sys

# mujoco settles its OpenGL backend when it is first imported, from MUJOCO_GL, and takes GLFW when
# that is unset. CrazyFlow imports mujoco, so the line below is that first import for everything
# under drones.sim, a console script's `main()` included: choosing there is too late. GLFW needs a
# display, so without one (a cluster node, ssh) offscreen rendering goes through EGL instead. With
# a display, or a MUJOCO_GL of the user's own, nothing changes.
if (sys.platform.startswith('linux') and not os.environ.get('DISPLAY')
        and not os.environ.get('WAYLAND_DISPLAY')):
    os.environ.setdefault('MUJOCO_GL', 'egl')

# CrazyFlow must be imported before anything pulls in scipy: it switches scipy into its array-API
# mode, which only takes effect on scipy's first import.
import crazyflow  # noqa: E402, F401
