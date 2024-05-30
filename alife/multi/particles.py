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
    0: ParticleCharachteristics(mass=1, radius=0.03),
    1: ParticleCharachteristics(mass=10, radius=0.06),
    2: ParticleCharachteristics(mass=100, radius=0.08),
}
P_CHARACHTERISTICS = ParticleCharachteristics(
    mass=jnp.array([v.mass for v in P_CHARACHTERISTICS.values()]),
    radius=jnp.array([v.radius for v in P_CHARACHTERISTICS.values()]),
)

# Postive values for the atomic repulsion coefficients will make the particles repel each other
ATOMIC_REPULSION_COEFFS = jnp.array(
    [
        [5, -10, -100],
        [-10, 10, -10],
        [-100, -10, 1000],
    ]
)
