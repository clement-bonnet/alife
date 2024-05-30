import time
from typing import Tuple, Callable

import jax.numpy as jnp
import jax
import chex

from alife.multi.visualizer import Visualizer
from alife.multi.particles import Particle, P_CHARACHTERISTICS
from alife.multi.forces import compute_forces, merge_particles
from alife.multi.elastic_collision import compute_elastic_collision_boundaries


def init_particles(
    key: chex.PRNGKey,
    max_num_particles: int = 128,
    p_particles: float = 0.8,
    padding: float = 0.1,
    speed_scaling: float = 10.0,
) -> Particle:
    positions_key, velocities_key, alive_key = jax.random.split(key, 3)
    positions = jax.random.uniform(
        positions_key, (max_num_particles, 2), minval=-1 + padding, maxval=1 - padding
    )
    velocities = speed_scaling * jax.random.normal(velocities_key, (max_num_particles, 2))
    types = jnp.zeros(max_num_particles, dtype=jnp.uint8)
    alives = jax.random.bernoulli(alive_key, p_particles, (max_num_particles,))
    particles = Particle(
        xy=positions,
        v=velocities,
        type=types,
        alive=alives,
    )
    return particles


def make_update_particles(dt: float, num_updates: int) -> Callable[[Particle], Particle]:
    def update_particles_once(particles: Particle, _) -> Tuple[Particle, None]:
        particles = merge_particles(particles)
        forces = compute_forces(particles)
        masses = jnp.take(P_CHARACHTERISTICS.mass, particles.type)
        accelerations = forces / masses[:, None]
        velocities = particles.v + dt * accelerations
        velocities = compute_elastic_collision_boundaries(velocities, particles)
        # Integration step
        positions = particles.xy + dt * velocities
        particles = Particle(
            xy=positions,
            v=velocities,
            type=particles.type,
            alive=particles.alive,
        )
        return particles, None

    def update_particles(particles: Particle) -> Particle:
        return jax.lax.scan(update_particles_once, particles, None, length=num_updates)[0]

    return jax.jit(update_particles)


def run():
    dt = 0.00001
    pause = 0.005
    plot_frequency = 2000
    num_steps = 5000000
    seed = 0

    particles = init_particles(jax.random.PRNGKey(seed), max_num_particles=32)
    visualizer = Visualizer()
    update_particles = make_update_particles(dt, plot_frequency)
    for step in range(num_steps // plot_frequency):
        t_0 = time.perf_counter()
        particles = update_particles(particles)
        fps = plot_frequency / (time.perf_counter() - t_0)
        visualizer.update_fig(particles, step * plot_frequency, fps, pause=pause)


if __name__ == "__main__":
    run()
