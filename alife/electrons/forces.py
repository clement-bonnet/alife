import jax
import jax.numpy as jnp

from alife.electrons.particles import Particle, P_CHARACHTERISTICS


def compute_forces(nuclei: Particle, electrons: Particle) -> tuple[jax.Array, jax.Array]:
    f_nuclei, f_electrons = n_e_force(nuclei, electrons)
    f_nuclei += repulsion_force(nuclei, P_CHARACHTERISTICS.radius[0])
    f_electrons += repulsion_force(electrons, P_CHARACHTERISTICS.radius[1])
    f_nuclei += friction(nuclei)
    f_electrons += friction(electrons)
    return f_nuclei, f_electrons


def repulsion_force(particle: Particle, radius: float, num_radii_characteristic: float = 2.0) -> jax.Array:
    coeff = -1.0
    differences = particle.xy[None, :] - particle.xy[:, None]
    distances = jnp.sqrt(jnp.sum(differences**2, axis=-1))
    normed_differences = differences / (distances[:, :, None] + 1e-6)
    radius_sum = radius + radius
    characteristic_distance = distances / (num_radii_characteristic * radius_sum)
    forces_p_to_p = coeff * normed_differences / (characteristic_distance[:, :, None] ** 7 + 1e-7)
    forces_p_to_p = jnp.where(
        (particle.alive[None, :] & particle.alive[:, None])[:, :, None], forces_p_to_p, 0
    )
    forces = jnp.sum(forces_p_to_p, axis=1)
    return forces


def n_e_force(nuclei: Particle, electrons: Particle) -> tuple[jax.Array, jax.Array]:
    coeff = 5000.0
    differences = nuclei.xy[None, :] - electrons.xy[:, None]
    distances = jnp.sqrt(jnp.sum(differences**2, axis=-1))
    normed_differences = differences / (distances[:, :, None] + 1e-6)
    radius_sum = P_CHARACHTERISTICS.radius.sum()
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
        (nuclei.alive[None, :] & electrons.alive[:, None])[:, :, None], forces_p_to_p, 0
    )
    nuclei_forces = jnp.sum(forces_p_to_p, axis=0)
    electrons_forces = jnp.sum(-forces_p_to_p, axis=1)
    return nuclei_forces, electrons_forces


def friction(particles: Particle) -> jax.Array:
    """Friction force is proportional to the squared speed of the particle."""
    friction_coefficient = 0.1
    forces = -friction_coefficient * particles.v * jnp.linalg.norm(particles.v, axis=-1, keepdims=True)
    forces = jnp.where(particles.alive[:, None], forces, 0)
    return forces
