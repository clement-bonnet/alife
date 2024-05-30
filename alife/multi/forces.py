import jax
import jax.numpy as jnp

from alife.multi.particles import Particle, P_CHARACHTERISTICS, ATOMIC_REPULSION_COEFFS


def compute_forces(particles: Particle) -> jax.Array:
    forces = jnp.zeros_like(particles.xy)
    forces += atomic_repulsion(particles)
    forces += friction(particles)
    return forces


def friction(particles: Particle) -> jax.Array:
    friction_coefficient = 1.0
    forces = -friction_coefficient * particles.v
    return forces


def atomic_repulsion(particles: Particle) -> jax.Array:
    # TODO: improve the curve shape of the atomic repulsion (more localized, take radius into account, etc.)
    differences = particles.xy[:, None] - particles.xy[None, :]
    distances = jnp.sqrt(jnp.sum(differences**2, axis=-1, keepdims=True))
    coeffs = ATOMIC_REPULSION_COEFFS[particles.type[:, None], particles.type[None, :], None]
    forces = jnp.sum(coeffs * differences / (distances**3 + 1e-3), axis=1)
    return forces


def merge_particles(particles: Particle) -> Particle:
    """TODO: right now, if there are triplets of particles, there might be multiple merges when only one would be allowed."""
    proximity_center_for_merge = 0.2
    differences = particles.xy[:, None] - particles.xy[None, :]
    distances = jnp.sqrt(jnp.sum(differences**2, axis=-1))
    # set lower triangle and diagonal to infinity
    distances = distances.at[jnp.tril_indices(distances.shape[0])].set(jnp.finfo(jnp.float32).max)
    # set distances of particles that are not alive to inf
    distances = jnp.where(
        particles.alive[:, None] @ particles.alive[None, :], distances, jnp.finfo(jnp.float32).max
    )
    # for each particle, find the closest other particle
    closest_indices = jnp.argmin(distances, axis=1)
    # for each particle, if the closest particle is within a range of 0.1 then merge them
    closest_particles_xy = jnp.take(particles.xy, closest_indices, axis=0)
    closest_particles_v = jnp.take(particles.v, closest_indices, axis=0)
    closest_particles_type = jnp.take(particles.type, closest_indices)
    closest_particles_alive = jnp.take(particles.alive, closest_indices)
    closest_particles_distance = distances[jnp.arange(distances.shape[0]), closest_indices]
    particles_radii = jnp.take(P_CHARACHTERISTICS.radius, particles.type)
    num_types = len(P_CHARACHTERISTICS.radius)
    merge_indices = (
        particles.alive
        & closest_particles_alive
        & (particles.type == closest_particles_type)
        & (particles.type < num_types - 1)
        & (closest_particles_distance < proximity_center_for_merge * particles_radii)
    )
    # make not alive the closest particle
    alive = particles.alive
    for i in range(len(merge_indices)):
        alive = alive.at[closest_indices[i]].set(
            jnp.where(merge_indices[i], False, alive[closest_indices[i]])
        )
    merged_x = (particles.xy + closest_particles_xy) / 2
    # the new speed is the speed that conserves the kinetic energy
    # 1/2 m_new v_new^2 = 1/2 m v^2 + 1/2 m v_closest^2
    # v_new = sqrt(2*m/m_new * (v^2 + v_closest^2))
    merged_v = (
        jnp.sqrt(
            2
            * jnp.take(P_CHARACHTERISTICS.mass, particles.type)
            / jnp.take(P_CHARACHTERISTICS.mass, particles.type + 1)
            * (jnp.sum(particles.v**2, axis=-1) + jnp.sum(closest_particles_v**2, axis=-1))
        )[:, None]
        * particles.v
        / (jnp.linalg.norm(particles.v, axis=-1, keepdims=True) + 1e-5)
    )
    particles = Particle(
        xy=jnp.where(merge_indices[:, None], merged_x, particles.xy),
        v=jnp.where(merge_indices[:, None], merged_v, particles.v),
        type=jnp.where(merge_indices, particles.type + 1, particles.type),
        alive=alive,
    )
    return particles
