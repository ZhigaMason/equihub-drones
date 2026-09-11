"""Guards on the package layout that simulator and RL work will depend on."""
import subprocess
import sys

FORBIDDEN = ('cflib', 'fastapi', 'uvicorn', 'starlette', 'websockets')


def test_control_package_never_imports_hardware_or_web_stacks():
    # The control law must run on a training box without the Crazyflie stack.
    # A fresh interpreter, so imports made by other tests cannot mask this.
    code = (
        'import sys\n'
        'import drones.control.avoidance, drones.control.mixer, '
        'drones.control.safety\n'
        f'bad = sorted(m for m in sys.modules if m.split(".")[0] in {FORBIDDEN!r})\n'
        'print(bad)\n'
        'sys.exit(1 if bad else 0)\n'
    )
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True)
    assert result.returncode == 0, (
        f'drones.control pulled in {result.stdout.strip()} {result.stderr}')


def test_flight_side_of_a_policy_needs_neither_jax_nor_the_simulator():
    # The laptop that flies a trained policy has only the base install: cflib and numpy.
    code = (
        'import sys\n'
        'import drones.policy.interface, drones.policy.runtime, drones.policy.square\n'
        'import drones.missions.fly_policy, drones.missions.fly_square\n'
        'bad = sorted(m for m in sys.modules if m.split(".")[0] in '
        '("jax", "flax", "optax", "crazyflow", "mujoco"))\n'
        'print(bad)\n'
        'sys.exit(1 if bad else 0)\n'
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, f'flight code pulled in {result.stdout.strip()} {result.stderr}'


def test_config_looks_for_env_at_the_repository_root():
    # A notebook or REPL opened outside the repo must still pick up the repo's
    # .env, so the path comes from the package's location, not the cwd.
    from drones import config
    assert (config._REPO_ENV.parent / 'pyproject.toml').is_file()
