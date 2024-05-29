import jax
import jax.numpy as jnp

from alife.multi.particles import Particle


def compute_forces(particles: Particle) -> jax.Array:
    forces = jnp.zeros_like(particles.xy)
    forces += boundary_force(particles)
    return forces


def boundary_force(particles: Particle) -> jax.Array:
    constant = 100.0
    power = 4
    forces = -constant * power * particles.xy ** (power - 1)
    return forces


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
