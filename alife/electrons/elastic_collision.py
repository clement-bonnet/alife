import jax
import jax.numpy as jnp

from alife.electrons.particles import P_CHARACHTERISTICS


def compute_elastic_collision_boundaries(
    v_nuclei: jax.Array,
    v_electrons: jax.Array,
    xy_nuclei: jax.Array,
    xy_electrons: jax.Array,
    min_grid_size: float,
    max_grid_size: float,
) -> jax.Array:
    nuclei_radius, electrons_nuclei = P_CHARACHTERISTICS.radius
    v_nuclei = update_velocity_particle(v_nuclei, xy_nuclei, nuclei_radius, min_grid_size, max_grid_size)
    v_electrons = update_velocity_particle(
        v_electrons, xy_electrons, electrons_nuclei, min_grid_size, max_grid_size
    )
    return v_nuclei, v_electrons


def update_velocity_particle(
    velocities: jax.Array, positions: jax.Array, radius: float, min_grid_size: float, max_grid_size: float
) -> jax.Array:
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
