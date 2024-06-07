from typing import NamedTuple

import jax


class Particle(NamedTuple):
    xy: jax.Array  # (2,)
    v: jax.Array  # (2,)
    genome: jax.Array  # (G,)
    alive: bool  # True, False


class ParticleCharachteristics(NamedTuple):
    radius: int
    mass: float


P_CHARACHTERISTICS = ParticleCharachteristics(mass=1, radius=0.04)
