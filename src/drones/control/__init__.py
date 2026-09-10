"""Hardware-free flight control law, shared by the real drone, simulators and RL.

Must never import cflib or any web framework: tests/test_architecture.py
enforces it.
"""
