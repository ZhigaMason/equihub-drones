"""The residual wrench: zero until fitted, body-frame in, world-frame out."""
import math

import pytest

pytest.importorskip('crazyflow')

import jax
import jax.numpy as jnp
import numpy as np

from drones.sim.residual import (FORCE_SCALE, Residual, init_residual, residual_features,
                                 residual_wrench)

HALF = math.sqrt(0.5)
YAWED_LEFT = jnp.array([[0.0, 0.0, HALF, HALF]])   # facing +y


def with_output_bias(params, bias):
    def set_bias(path, x):
        name = jax.tree_util.keystr(path)
        return jnp.asarray(bias, x.dtype) if "'out'" in name and "'bias'" in name else x
    return jax.tree_util.tree_map_with_path(set_bias, params)


def test_an_unfitted_residual_changes_nothing():
    params = init_residual(jax.random.key(0))
    force, torque = residual_wrench(Residual(), params, jnp.ones((3, 3)),
                                    jnp.tile(YAWED_LEFT, (3, 1)), jnp.ones((3, 3)),
                                    jnp.ones((3, 4)))
    np.testing.assert_array_equal(force, 0.0)
    np.testing.assert_array_equal(torque, 0.0)


def test_body_frame_force_is_rotated_into_the_world():
    params = with_output_bias(init_residual(jax.random.key(0)), [1.0, 0, 0, 0, 0, 0])
    force, _ = residual_wrench(Residual(), params, jnp.zeros((1, 3)), YAWED_LEFT,
                               jnp.zeros((1, 3)), jnp.zeros((1, 4)))
    np.testing.assert_allclose(force[0], [0.0, FORCE_SCALE, 0.0], atol=1e-6)


def test_features_are_in_the_body_frame():
    features = residual_features(jnp.array([[0.0, 1.0, 0.0]]), YAWED_LEFT, jnp.zeros((1, 3)),
                                 jnp.zeros((1, 4)))
    np.testing.assert_allclose(features[0, :3], [1.0, 0.0, 0.0], atol=1e-6)   # moving forward
    np.testing.assert_allclose(features[0, 3:6], [0.0, 0.0, -1.0], atol=1e-6)  # level
