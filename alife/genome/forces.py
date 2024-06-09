import jax
import jax.numpy as jnp

from alife.genome.particles import Particle, P_CHARACHTERISTICS


def compute_forces(particles: Particle, weights: jax.Array) -> jax.Array:
    f_particles = p2p_force(particles, weights)
    f_particles += friction(particles)
    return f_particles


def p2p_force(particles: Particle, weights: jax.Array) -> jax.Array:
    def force_distance(distance, force_weights, genome):
        coeffs = jnp.dot(force_weights.T, genome)
        force = jnp.dot(jnp.stack([distance**i for i in range(force_weights.shape[-1])], axis=-1), coeffs)
        force *= jnp.exp(-1.3 * distance)
        return force

    differences = particles.xy[None, :] - particles.xy[:, None]
    distances = jnp.sqrt(jnp.sum(differences**2, axis=-1))
    force_direction = differences / (distances[:, :, None] + 1e-6)
    radius_sum = 2 * P_CHARACHTERISTICS.radius
    characteristic_distance = distances[:, :, None] / radius_sum
    interaction_strength = jax.vmap(force_distance, in_axes=(0, None, 0))(
        characteristic_distance, weights, particles.genome
    )
    forces_p_to_p = force_direction * interaction_strength
    forces_p_to_p = jnp.where(
        (particles.alive[None, :, None] & particles.alive[:, None, None]), forces_p_to_p, 0
    )
    particles_forces = jnp.sum(forces_p_to_p, axis=0)
    return particles_forces


def friction(particles: Particle) -> jax.Array:
    """Friction force is proportional to the squared speed of the particle."""
    friction_coefficient = 0.05
    forces = -friction_coefficient * particles.v * jnp.linalg.norm(particles.v, axis=-1, keepdims=True)
    forces = jnp.where(particles.alive[:, None], forces, 0)
    return forces


def increase_velocities(velocities: jax.Array, energy: float) -> jax.Array:
    speed = jnp.sqrt(jnp.sum(velocities**2, axis=-1, keepdims=True))
    velocities_normalized = velocities / (speed + 1e-6)
    # v2**2 = v1**2 + 2*energy/mass
    return velocities_normalized * jnp.sqrt(speed**2 + 2 * energy / P_CHARACHTERISTICS.mass)


def particles_capture_energy(
    xy_particles: jax.Array,
    v_particles: jax.Array,
    min_grid_size: float,
    energy_source_size: float,
    energy_coeff: float,
    nrg_y: float,
) -> tuple[jax.Array, jax.Array]:
    v_particles = jnp.where(
        jnp.expand_dims(
            (xy_particles[:, 0] <= min_grid_size + energy_source_size)
            & (xy_particles[:, 1] >= nrg_y)
            & (xy_particles[:, 1] <= nrg_y + energy_source_size),
            axis=1,
        ),
        increase_velocities(v_particles, energy_coeff),
        v_particles,
    )
    return v_particles
