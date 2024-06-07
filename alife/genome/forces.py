import jax
import jax.numpy as jnp

from alife.genome.particles import Particle, P_CHARACHTERISTICS


def compute_forces(particles: Particle) -> tuple[jax.Array, jax.Array]:
    f_particles = p2p_force(particles)
    f_particles += friction(particles)
    return f_particles


def p2p_force(particles: Particle) -> jax.Array:
    # TODO: implement the genome-specific forces
    coeff = 50.0
    differences = particles.xy[None, :] - particles.xy[:, None]
    distances = jnp.sqrt(jnp.sum(differences**2, axis=-1))
    normed_differences = differences / (distances[:, :, None] + 1e-6)
    radius_sum = 2 * P_CHARACHTERISTICS.radius
    characteristic_distance = distances / (0.5 * radius_sum)
    # Energy potential: 1/((x/0.8)^6 + 1/10) - 5*exp(-(x-1)^2/0.3^2) + 4*exp(-(x-1.5)^2/0.3^2)
    x = characteristic_distance[:, :, None]
    # -89 e^(-11 (-1.5 + x)^2) (-1.5 + x) + 111 e^(-11 (-1 + x)^2) (-1 + x) - (1.57 x^5)/(0.03 + x^6)^2
    forces_p_to_p = (
        coeff * normed_differences * -89 * jnp.exp(-11 * (-1.5 + x) ** 2) * (-1.5 + x)
        + 111 * jnp.exp(-11 * (-1 + x) ** 2) * (-1 + x)
        - (1.57 * x**5) / (0.03 + x**6) ** 2
    )
    forces_p_to_p = jnp.where(
        (particles.alive[None, :] & particles.alive[:, None])[:, :, None], forces_p_to_p, 0
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
