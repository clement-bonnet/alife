import jax
import jax.numpy as jnp

from alife.multi.particles import Particle, P_CHARACHTERISTICS


def compute_forces(particles: Particle) -> jax.Array:
    forces = jnp.zeros_like(particles.xy)
    forces += atomic_repulsion(particles)
    return forces


def atomic_repulsion(particles: Particle) -> jax.Array:
    # TODO: improve the curve shape of the atomic repulsion (more localized, take radius into account, etc.)
    constant = 5e0
    differences = particles.xy[:, None] - particles.xy[None, :]
    distances = jnp.sqrt(jnp.sum(differences**2, axis=-1, keepdims=True))
    forces = constant * jnp.sum(differences / (distances**3 + 1e-9), axis=1)
    return forces


def merge_particles(particles: Particle) -> Particle:
    differences = particles.xy[:, None] - particles.xy[None, :]
    distances = jnp.sqrt(jnp.sum(differences**2, axis=-1))
    # set distances to itself to infinity
    distances = distances.at[jnp.arange(distances.shape[0]), jnp.arange(distances.shape[0])].set(jnp.inf)
    # set lower triangle to infinity
    distances = distances.at[jnp.tril_indices(distances.shape[0])].set(jnp.inf)
    # for each particle, find the closest other particle
    closest_indices = jnp.argmin(distances, axis=1)
    # for each particle, if the closest particle is within a range of 0.1 then merge them
    closest_particles_xy = jnp.take(particles.xy, closest_indices, axis=0)
    closest_particles_v = jnp.take(particles.v, closest_indices, axis=0)
    closest_particles_type = jnp.take(particles.type, closest_indices)
    closest_particles_alive = jnp.take(particles.alive, closest_indices)
    closest_particles_distance = jnp.take(distances, closest_indices)
    particles_radii = jnp.take(P_CHARACHTERISTICS.radius, particles.type)
    num_types = len(P_CHARACHTERISTICS.radius)
    # maybe distances[jnp.arange(distances.shape[0]), closest_indices]
    merge_indices = (
        particles.alive
        & closest_particles_alive
        & (particles.type == closest_particles_type)
        & (particles.type < num_types - 1)
        & (closest_particles_distance < 10 * particles_radii)
    )
    # make not alive the closest particle
    # alive = jnp.where(merge_indices, False, particles.alive)  # TODO: I think this is wrong
    alive = particles.alive
    particles = Particle(
        xy=jnp.where(merge_indices[:, None], (particles.xy + closest_particles_xy) / 2, particles.xy),
        v=jnp.where(merge_indices[:, None], (particles.v + closest_particles_v) / 2, particles.v),
        type=jnp.where(merge_indices, particles.type + 1, particles.type),
        alive=alive,
    )
    return particles


# def compute_forces(positions, masses, radii):
#     num_particles = positions.shape[0]
#     forces = jnp.zeros_like(positions)
#     G = 6.67e-11

#     def compute_force_ij(forces, distance, diff, i, j):
#         force_magnitude = G * masses[i] * masses[j] / (distance**2)
#         force_vector = force_magnitude * diff / distance
#         forces = forces.at[i].set(force_vector)
#         forces = forces.at[j].set(-force_vector)
#         return forces

#     for i in range(num_particles):
#         for j in range(i + 1, num_particles):
#             diff = positions[j] - positions[i]
#             distance = jnp.sqrt(jnp.sum(diff**2))
#             is_colliding = distance < radii[i] + radii[j]
#             # Skip gravitational force calculation if colliding
#             forces = jax.lax.cond(
#                 is_colliding,
#                 lambda f: f,
#                 lambda f: compute_force_ij(f, distance, diff, i, j),
#                 forces,
#             )
#     return forces


# def resolve_collisions(positions, velocities, masses, radii):
#     def resolve_collision_ij(velocities, i, j):
#         # Normal and tangential decomposition of velocities
#         normal = diff / distance
#         v_i_normal = jnp.dot(velocities[i], normal) * normal
#         v_j_normal = jnp.dot(velocities[j], normal) * normal
#         v_i_tangent = velocities[i] - v_i_normal
#         v_j_tangent = velocities[j] - v_j_normal

#         # Update velocities based on conservation of momentum and kinetic energy
#         velocities = velocities.at[i].set(v_j_normal + v_i_tangent)
#         velocities = velocities.at[j].set(v_i_normal + v_j_tangent)
#         return velocities

#     num_particles = positions.shape[0]
#     for i in range(num_particles):
#         for j in range(i + 1, num_particles):
#             diff = positions[j] - positions[i]
#             distance = jnp.sqrt(jnp.sum(diff**2))
#             # Check for collision
#             velocities = jax.lax.cond(
#                 distance < radii[i] + radii[j],
#                 lambda v: resolve_collision_ij(v, i, j),
#                 lambda v: v,
#                 velocities,
#             )
#     return velocities


# def resolve_boundary_collisions(positions, velocities, radii):
#     velocities = velocities.at[:, 0].set(
#         jnp.where(positions[:, 0] - radii <= 0, jnp.abs(velocities[:, 0]), velocities[:, 0])
#     )
#     velocities = velocities.at[:, 0].set(
#         jnp.where(positions[:, 0] + radii >= 1, -jnp.abs(velocities[:, 0]), velocities[:, 0])
#     )
#     velocities = velocities.at[:, 1].set(
#         jnp.where(positions[:, 1] - radii <= 0, jnp.abs(velocities[:, 1]), velocities[:, 1])
#     )
#     velocities = velocities.at[:, 1].set(
#         jnp.where(positions[:, 1] + radii >= 1, -jnp.abs(velocities[:, 1]), velocities[:, 1])
#     )
#     return velocities
