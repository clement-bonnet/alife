import time
from typing import Tuple, Callable

import jax.numpy as jnp
import jax

jax.config.update("jax_platform_name", "cpu")

import chex

from alife.genome.visualizer import Visualizer
from alife.genome.particles import Particle, P_CHARACHTERISTICS
from alife.genome.forces import compute_forces, particles_capture_energy
from alife.genome.elastic_collision import (
    compute_elastic_collision_boundaries,
    compute_elastic_collision_wall,
)


def init_particles(
    key: chex.PRNGKey,
    num_particles: int = 4,
    speed_scaling: float = 5.0,
    padding: float = 0.1,
    min_grid_size: float = -1.0,
    max_grid_size: float = 1.0,
    genome_length: int = 4,
) -> tuple[Particle, tuple[float, float]]:
    positions_key, velocities_key, genome_key = jax.random.split(key, 3)
    particles = Particle(
        xy=jax.random.uniform(
            positions_key,
            (num_particles, 2),
            minval=min_grid_size + padding,
            maxval=max_grid_size - padding,
        ),
        v=speed_scaling * jax.random.normal(velocities_key, (num_particles, 2)),
        genome=jax.random.bernoulli(genome_key, 0.5, (num_particles, genome_length)),
        alive=jnp.ones(num_particles, dtype=bool),
    )
    energy_source = (min_grid_size, 1.0)
    return particles, energy_source


def make_update_particles(
    dt: float,
    num_updates: int,
    force_weights: jax.Array,
    min_grid_size: float = -1.0,
    max_grid_size: float = 1.0,
    wall: bool = False,
    wall_gap_size: float = 0.5,
    energy_source_bool: bool = False,
    energy_coeff: float = 0.1,
    energy_source_speed: float = 0.1,
    energy_source_size: float = 0.5,
    friction_coefficient: float = 0.05,
    force_scaling: float = 10,
) -> Callable[[Particle, tuple[float, float]], tuple[Particle, tuple[float, float]]]:
    def update_particles_once(
        particles: Particle, energy_source: tuple[float, float]
    ) -> Tuple[Particle, tuple[float, float]]:
        f_particles = compute_forces(particles, force_weights, friction_coefficient, force_scaling)
        a_particles = f_particles / P_CHARACHTERISTICS.mass
        v_particles = particles.v + dt * a_particles
        v_particles = compute_elastic_collision_boundaries(
            v_particles, particles.xy, min_grid_size, max_grid_size
        )
        if wall:
            v_particles = compute_elastic_collision_wall(v_particles, particles.xy, wall_gap_size)
        if energy_source_bool:
            nrg_y, nrg_v_sign = energy_source
            nrg_v_sign = jnp.where(
                (nrg_y + energy_source_size > max_grid_size) | (nrg_y < min_grid_size),
                -nrg_v_sign,
                nrg_v_sign,
            )
            nrg_y += dt * nrg_v_sign * energy_source_speed
            # Add energy to the particles
            v_particles = particles_capture_energy(
                particles.xy,
                v_particles,
                min_grid_size,
                energy_source_size,
                energy_coeff,
                nrg_y,
            )
            energy_source = (nrg_y, nrg_v_sign)

        # Integration step
        xy_particles = particles.xy + dt * v_particles
        particles = Particle(
            xy=xy_particles,
            v=v_particles,
            genome=particles.genome,
            alive=particles.alive,
        )
        return particles, energy_source

    def update_particles(
        particles: Particle, energy_source: tuple[float, float]
    ) -> tuple[Particle, tuple[float, float]]:
        return jax.lax.scan(
            lambda args, _: (update_particles_once(*args), None),
            (particles, energy_source),
            None,
            length=num_updates,
        )[0]

    return jax.jit(update_particles)


def run():
    dt = 0.001
    pause = 0.04
    plot_frequency = 500
    num_steps = 20_000_000
    grid_size = [-3.0, 3.0]
    wall = True
    wall_gap_size = 1.0
    energy_source_bool = False
    energy_coeff = 0.0005
    energy_source_speed = 0.05
    energy_source_size = 1.0
    friction_coefficient = 0.1
    force_scaling = 10
    num_particles = 128
    genome_length = 8
    num_coefficients_forces = 8
    seed = 0

    key_particles, key_forces = jax.random.split(jax.random.PRNGKey(seed))
    particles, energy_source = init_particles(
        key_particles,
        num_particles=num_particles,
        speed_scaling=1.0,
        min_grid_size=grid_size[0],
        max_grid_size=grid_size[1],
        genome_length=genome_length,
    )
    visualizer = Visualizer(*grid_size, wall, wall_gap_size, energy_source_bool, energy_source_size)

    scales = 1 / 10 ** (jnp.arange(num_coefficients_forces) / 2)
    W = scales[None, :] * jax.random.normal(key_forces, (genome_length, num_coefficients_forces))
    update_particles = make_update_particles(
        dt,
        plot_frequency,
        W,
        *grid_size,
        wall,
        wall_gap_size,
        energy_source_bool,
        energy_coeff,
        energy_source_speed,
        energy_source_size,
        friction_coefficient,
        force_scaling,
    )
    print("Device:", particles.xy.devices())
    for step in range(num_steps // plot_frequency):
        t_0 = time.perf_counter()
        # particles, energy_source = jax.block_until_ready(update_particles(particles, energy_source))
        particles, energy_source = update_particles(particles, energy_source)
        fps = plot_frequency / (time.perf_counter() - t_0)
        visualizer.update_fig(particles, energy_source, step * plot_frequency, fps, pause=pause)


if __name__ == "__main__":
    run()
    # TODO: improve visualization speed
