"""Make the suite hermetic before anything imports drones.config.

Tests assert behaviour at the tuning shipped in code. A local .env holds
operator tuning (git-ignored, different per drone), so loading it would make
results depend on whose machine runs them.
"""
import os

os.environ['DRONES_NO_DOTENV'] = '1'

# CrazyFlow refuses to import once scipy has been loaded without its array-API mode, and cflib
# imports scipy. Setting the flag before any test module imports anything keeps both working.
os.environ.setdefault('SCIPY_ARRAY_API', '1')
