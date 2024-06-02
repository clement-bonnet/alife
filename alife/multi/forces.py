import jax
import jax.numpy as jnp

from alife.multi.particles import Particle, P_CHARACHTERISTICS, ATOMIC_REPULSION_COEFFS


def compute_forces(particles: Particle) -> jax.Array:
    forces = atomic_repulsion(particles)
    forces += friction(particles)
    return forces


def friction(particles: Particle) -> jax.Array:
    """Friction force is proportional to the squared speed of the particle."""
    friction_coefficient = 0.01
    forces = -friction_coefficient * particles.v * jnp.linalg.norm(particles.v, axis=-1, keepdims=True)
    forces = jnp.where(particles.alive[:, None], forces, 0)
    return forces


def atomic_repulsion(particles: Particle) -> jax.Array:
    # TODO: improve the curve shape of the atomic repulsion (more localized, take radius into account, etc.)
    differences = particles.xy[None, :] - particles.xy[:, None]
    distances = jnp.sqrt(jnp.sum(differences**2, axis=-1))
    radii = jnp.take(P_CHARACHTERISTICS.radius, particles.type)
    radii_sum = radii[:, None] + radii[None, :]
    characteristic_distance = distances / (2 * radii_sum)
    coeffs = ATOMIC_REPULSION_COEFFS[particles.type[:, None], particles.type[None, :], None]
    forces_p_to_p = coeffs * differences / (characteristic_distance[:, :, None] ** 7 + 1e-6)
    forces_p_to_p = jnp.where(
        (particles.alive[:, None] & particles.alive[None, :])[:, :, None], forces_p_to_p, 0
    )
    forces = jnp.sum(forces_p_to_p, axis=1)
    return forces


def merge_particles(particles: Particle) -> Particle:
    """TODO: right now, if there are triplets of particles, there might be multiple merges when only one would be allowed."""
    proximity_center_for_merge = 0.5
    differences = particles.xy[None, :] - particles.xy[:, None]
    distances = jnp.sqrt(jnp.sum(differences**2, axis=-1))
    # set lower triangle and diagonal to infinity
    distances = distances.at[jnp.tril_indices(distances.shape[0])].set(jnp.finfo(jnp.float32).max)
    # set distances of particles that are not alive to inf
    distances = jnp.where(
        particles.alive[:, None] & particles.alive[None, :], distances, jnp.finfo(jnp.float32).max
    )
    # for each particle, find the closest other particle
    closest_indices = jnp.argmin(distances, axis=1)
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
    # closest_indices = jnp.where(merge_indices, closest_indices, jnp.arange(len(particles.xy), 2*len(particles.xy)))
    # alive = alive.at[closest_indices].set(jnp.where(merge_indices, False, alive[closest_indices]))
    # TODO: improve this loop, SLOW!!
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
