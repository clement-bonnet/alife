import jax
import jax.numpy as jnp

from alife.multi.particles import Particle, P_CHARACHTERISTICS


def compute_elastic_collision_boundaries(velocities: jax.Array, particles: Particle) -> jax.Array:
    radii = jnp.take(P_CHARACHTERISTICS.radius, particles.type)
    # TODO: try merge these operations into one
    velocities = velocities.at[:, 0].set(
        jnp.where(particles.xy[:, 0] - radii <= -1, jnp.abs(velocities[:, 0]), velocities[:, 0])
    )
    velocities = velocities.at[:, 0].set(
        jnp.where(particles.xy[:, 0] + radii >= 1, -jnp.abs(velocities[:, 0]), velocities[:, 0])
    )
    velocities = velocities.at[:, 1].set(
        jnp.where(particles.xy[:, 1] - radii <= -1, jnp.abs(velocities[:, 1]), velocities[:, 1])
    )
    velocities = velocities.at[:, 1].set(
        jnp.where(particles.xy[:, 1] + radii >= 1, -jnp.abs(velocities[:, 1]), velocities[:, 1])
    )
    return velocities
