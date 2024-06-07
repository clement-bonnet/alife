import jax
import jax.numpy as jnp

from alife.genome.particles import P_CHARACHTERISTICS


def compute_elastic_collision_boundaries(
    velocities: jax.Array,
    positions: jax.Array,
    min_grid_size: float,
    max_grid_size: float,
) -> jax.Array:
    radius = P_CHARACHTERISTICS.radius
    # TODO: try merge these operations into one
    velocities = velocities.at[:, 0].set(
        jnp.where(positions[:, 0] - radius <= min_grid_size, jnp.abs(velocities[:, 0]), velocities[:, 0])
    )
    velocities = velocities.at[:, 0].set(
        jnp.where(positions[:, 0] + radius >= max_grid_size, -jnp.abs(velocities[:, 0]), velocities[:, 0])
    )
    velocities = velocities.at[:, 1].set(
        jnp.where(positions[:, 1] - radius <= min_grid_size, jnp.abs(velocities[:, 1]), velocities[:, 1])
    )
    velocities = velocities.at[:, 1].set(
        jnp.where(positions[:, 1] + radius >= max_grid_size, -jnp.abs(velocities[:, 1]), velocities[:, 1])
    )
    return velocities


def update_velocity_particle_boundaries(
    velocities: jax.Array, positions: jax.Array, radius: float, min_grid_size: float, max_grid_size: float
) -> jax.Array:
    return velocities


def compute_elastic_collision_wall(
    velocities: jax.Array,
    positions: jax.Array,
    wall_gap_size: float,
) -> jax.Array:
    # Reverse the y speed of the particles that hit the wall which is at y = 0
    v_signs = jnp.where(positions[:, 1] >= 0, 1, -1)
    x_conditions = (positions[:, 0] <= -wall_gap_size / 2) | (positions[:, 0] >= wall_gap_size / 2)
    radius = P_CHARACHTERISTICS.radius
    y_conditions = (-radius <= positions[:, 1]) & (positions[:, 1] <= radius)
    velocities = velocities.at[:, 1].set(
        jnp.where(
            x_conditions & y_conditions,
            v_signs * jnp.abs(velocities[:, 1]),
            velocities[:, 1],
        )
    )
    return velocities
