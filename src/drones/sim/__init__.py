"""CrazyFlow simulation of the Crazyflie 2.1 Brushless: scene, sensors and training tasks.

Needs the optional ``sim`` extra: ``uv sync --extra sim``.
"""
# CrazyFlow must be imported before anything pulls in scipy: it switches scipy into its array-API
# mode, which only takes effect on scipy's first import.
import crazyflow  # noqa: F401
