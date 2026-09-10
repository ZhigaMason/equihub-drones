"""Actor-critic networks for sensor-based policies.

The actor only sees what the real drone can sense: stacked sensor frames and, when the camera is
on, an image. The critic is only used in training, so it also gets privileged simulator state
(position in the room, true velocity). That sharpens value estimates without the deployed policy
depending on anything it cannot measure.
"""
import flax.linen as nn
import jax.numpy as jnp


class ImageEncoder(nn.Module):
    """A small strided CNN; cheap enough to run on every step of every world."""
    features: int = 64

    @nn.compact
    def __call__(self, image):
        x = image
        for channels in (16, 32, 32):
            x = nn.relu(nn.Conv(channels, (3, 3), strides=(2, 2))(x))
        x = x.reshape(*x.shape[:-3], -1)
        return nn.relu(nn.Dense(self.features)(x))


class MLP(nn.Module):
    hidden: tuple[int, ...]
    out: int
    out_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        for size in self.hidden:
            x = nn.tanh(nn.Dense(size, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x))
        return nn.Dense(self.out, kernel_init=nn.initializers.orthogonal(self.out_scale))(x)


class ActorCritic(nn.Module):
    action_size: int
    hidden: tuple[int, ...] = (256, 256)
    use_image: bool = False

    def setup(self):
        # A small output scale starts the policy near zero action, which is calibrated hover.
        self.actor = MLP(self.hidden, self.action_size, out_scale=0.01)
        self.critic = MLP(self.hidden, 1, out_scale=1.0)
        if self.use_image:
            self.actor_image = ImageEncoder()
            self.critic_image = ImageEncoder()
        # Initial exploration std of 0.37 in normalised units: about +/-7 degrees of attitude noise.
        self.log_std = self.param('log_std', nn.initializers.constant(-1.0), (self.action_size,))

    def __call__(self, obs):
        """(action mean, log std, value) for a batch of observations."""
        return self.act(obs), self.log_std, self.value(obs)

    def act(self, obs):
        """Deterministic action. Reads only obs['policy'] (and obs['image']): deployable."""
        x = obs['policy']
        if self.use_image:
            x = jnp.concatenate([x, self.actor_image(obs['image'])], -1)
        return self.actor(x)

    def value(self, obs):
        x = obs['critic']
        if self.use_image:
            x = jnp.concatenate([x, self.critic_image(obs['image'])], -1)
        return self.critic(x)[..., 0]
