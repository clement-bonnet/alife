from typing import NamedTuple

import jax.numpy as jnp
import jax


class Particle(NamedTuple):
    xy: jax.Array  # (2,)
    v: jax.Array  # (2,)
    type: int  # 0, 1, 2
    alive: bool  # True, False


class ParticleCharachteristics(NamedTuple):
    radius: int
    mass: float


P_CHARACHTERISTICS = {
    "nucleus": ParticleCharachteristics(mass=10, radius=0.08),
    "electron": ParticleCharachteristics(mass=1, radius=0.04),
}
P_CHARACHTERISTICS = ParticleCharachteristics(
    mass=jnp.array([v.mass for v in P_CHARACHTERISTICS.values()]),
    radius=jnp.array([v.radius for v in P_CHARACHTERISTICS.values()]),
)
