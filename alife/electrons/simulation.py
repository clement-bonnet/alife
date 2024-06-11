import time
from typing import Tuple, Callable

import jax.numpy as jnp
import jax

jax.config.update("jax_platform_name", "cpu")

import chex

from alife.electrons.visualizer import Visualizer
from alife.electrons.particles import Particle, P_CHARACHTERISTICS
from alife.electrons.forces import compute_forces, particles_capture_energy
from alife.electrons.elastic_collision import (
    compute_elastic_collision_boundaries,
    compute_elastic_collision_wall,
)


def init_particles(
    key: chex.PRNGKey,
    num_nuclei: int = 4,
    num_electrons: int = 32,
    nuclei_speed_scaling: float = 8.0,
    electron_speed_scaling: float = 8.0,
    padding: float = 0.1,
    min_grid_size: float = -1.0,
    max_grid_size: float = 1.0,
) -> tuple[Particle, Particle, tuple[float, float]]:
    nuclei_key, electrons_key = jax.random.split(key)
    nuclei_positions_key, nuclei_velocities_key = jax.random.split(nuclei_key)
    nuclei = Particle(
        xy=jax.random.uniform(
            nuclei_positions_key,
            (num_nuclei, 2),
            minval=min_grid_size + padding,
            maxval=max_grid_size - padding,
        ),
        v=nuclei_speed_scaling * jax.random.normal(nuclei_velocities_key, (num_nuclei, 2)),
        type=jnp.zeros(num_nuclei, dtype=jnp.uint8),
        alive=jnp.ones(num_nuclei, dtype=bool),
    )
    elec_positions_key, elec_velocities_key = jax.random.split(electrons_key)
    electrons = Particle(
        xy=jax.random.uniform(
            elec_positions_key,
            (num_electrons, 2),
            minval=min_grid_size + padding,
            maxval=max_grid_size - padding,
        ),
        v=electron_speed_scaling * jax.random.normal(elec_velocities_key, (num_electrons, 2)),
        type=jnp.ones(num_electrons, dtype=jnp.uint8),
        alive=jnp.ones(num_electrons, dtype=bool),
    )
    energy_source = (min_grid_size, 1.0)
    return nuclei, electrons, energy_source


def make_update_particles(
    dt: float,
    num_updates: int,
    min_grid_size: float = -1.0,
    max_grid_size: float = 1.0,
    wall: bool = False,
    wall_gap_size: float = 0.5,
    energy_source_bool: bool = False,
    energy_coeff: float = 0.1,
    energy_source_speed: float = 0.1,
    energy_source_size: float = 0.5,
) -> Callable[[Particle, Particle, float], tuple[Particle, Particle, float]]:
    def update_particles_once(
        nuclei: Particle, electrons: Particle, energy_source: tuple[float, float]
    ) -> Tuple[Particle, Particle, tuple[float, float]]:
        f_nuclei, f_electrons = compute_forces(nuclei, electrons)
        m_nuclei, m_electrons = P_CHARACHTERISTICS.mass
        a_nuclei, a_electrons = f_nuclei / m_nuclei, f_electrons / m_electrons
        v_nuclei, v_electrons = nuclei.v + dt * a_nuclei, electrons.v + dt * a_electrons
        v_nuclei, v_electrons = compute_elastic_collision_boundaries(
            v_nuclei, v_electrons, nuclei.xy, electrons.xy, min_grid_size, max_grid_size
        )
        if wall:
            v_nuclei, v_electrons = compute_elastic_collision_wall(
                v_nuclei, v_electrons, nuclei.xy, electrons.xy, wall_gap_size
            )
        if energy_source_bool:
            nrg_y, nrg_v_sign = energy_source
            nrg_v_sign = jnp.where(
                (nrg_y + energy_source_size > max_grid_size) | (nrg_y < min_grid_size),
                -nrg_v_sign,
                nrg_v_sign,
            )
            nrg_y += dt * nrg_v_sign * energy_source_speed
            # Add energy to the particles
            v_nuclei, v_electrons = particles_capture_energy(
                nuclei.xy,
                electrons.xy,
                v_nuclei,
                v_electrons,
                min_grid_size,
                energy_source_size,
                energy_coeff,
                nrg_y,
            )

        # Integration step
        xy_nuclei, xy_electrons = nuclei.xy + dt * v_nuclei, electrons.xy + dt * v_electrons
        nuclei = Particle(
            xy=xy_nuclei,
            v=v_nuclei,
            type=nuclei.type,
            alive=nuclei.alive,
        )
        electrons = Particle(
            xy=xy_electrons,
            v=v_electrons,
            type=electrons.type,
            alive=electrons.alive,
        )
        return nuclei, electrons, (nrg_y, nrg_v_sign)

    def update_particles(
        nuclei: Particle, electrons: Particle, energy_source: tuple[float, float]
    ) -> tuple[Particle, Particle, tuple[float, float]]:
        return jax.lax.scan(
            lambda args, _: (update_particles_once(*args), None),
            (nuclei, electrons, energy_source),
            None,
            length=num_updates,
        )[0]

    return jax.jit(update_particles)


def run():
    dt = 0.0001
    pause = 0.1
    plot_frequency = 5000
    num_steps = 20_000_000
    grid_size = [-3.0, 3.0]
    wall = True
    wall_gap_size = 1.0
    energy_source_bool = True
    energy_coeff = 0.005
    energy_source_speed = 0.02
    energy_source_size = 1.0
    seed = 0

    nuclei, electrons, energy_source = init_particles(
        jax.random.PRNGKey(seed),
        num_nuclei=24,
        num_electrons=128,
        nuclei_speed_scaling=1.0,
        electron_speed_scaling=1.0,
        min_grid_size=grid_size[0],
        max_grid_size=grid_size[1],
    )
    visualizer = Visualizer(*grid_size, wall, wall_gap_size, energy_source_bool, energy_source_size)
    update_particles = make_update_particles(
        dt,
        plot_frequency,
        *grid_size,
        wall,
        wall_gap_size,
        energy_source_bool,
        energy_coeff,
        energy_source_speed,
        energy_source_size,
    )
    print("Device:", nuclei.xy.devices())
    for step in range(num_steps // plot_frequency):
        t_0 = time.perf_counter()
        nuclei, electrons, energy_source = jax.block_until_ready(
            update_particles(nuclei, electrons, energy_source)
        )
        fps = plot_frequency / (time.perf_counter() - t_0)
        visualizer.update_fig(nuclei, electrons, energy_source, step * plot_frequency, fps, pause=pause)


if __name__ == "__main__":
    run()
    # TODO: improve visualization speed (e.g. https://vispy.org/)
    # TODO: create a new project with 1 type of particles with a "genome" behavior that describes the interaction forces
