import time
from typing import Tuple, Callable

import jax.numpy as jnp
import jax

jax.config.update("jax_platform_name", "cpu")

import chex

from alife.electrons.visualizer import Visualizer
from alife.electrons.particles import Particle, P_CHARACHTERISTICS
from alife.electrons.forces import compute_forces
from alife.electrons.elastic_collision import compute_elastic_collision_boundaries


def init_particles(
    key: chex.PRNGKey,
    num_nuclei: int = 4,
    num_electrons: int = 32,
    nuclei_speed_scaling: float = 8.0,
    electron_speed_scaling: float = 8.0,
    padding: float = 0.1,
) -> tuple[Particle, Particle]:
    nuclei_key, electrons_key = jax.random.split(key)
    nuclei_positions_key, nuclei_velocities_key = jax.random.split(nuclei_key)
    nuclei = Particle(
        xy=jax.random.uniform(nuclei_positions_key, (num_nuclei, 2), minval=-1 + padding, maxval=1 - padding),
        v=nuclei_speed_scaling * jax.random.normal(nuclei_velocities_key, (num_nuclei, 2)),
        type=jnp.zeros(num_nuclei, dtype=jnp.uint8),
        alive=jnp.ones(num_nuclei, dtype=bool),
    )
    elec_positions_key, elec_velocities_key = jax.random.split(electrons_key)
    electrons = Particle(
        xy=jax.random.uniform(
            elec_positions_key, (num_electrons, 2), minval=-1 + padding, maxval=1 - padding
        ),
        v=electron_speed_scaling * jax.random.normal(elec_velocities_key, (num_electrons, 2)),
        type=jnp.ones(num_electrons, dtype=jnp.uint8),
        alive=jnp.ones(num_electrons, dtype=bool),
    )
    return nuclei, electrons


def make_update_particles(
    dt: float, num_updates: int
) -> Callable[[Particle, Particle], tuple[Particle, Particle]]:
    def update_particles_once(nuclei: Particle, electrons: Particle) -> Tuple[Particle, Particle]:
        f_nuclei, f_electrons = compute_forces(nuclei, electrons)
        m_nuclei, m_electrons = P_CHARACHTERISTICS.mass
        a_nuclei, a_electrons = f_nuclei / m_nuclei, f_electrons / m_electrons
        v_nuclei, v_electrons = nuclei.v + dt * a_nuclei, electrons.v + dt * a_electrons
        v_nuclei, v_electrons = compute_elastic_collision_boundaries(
            v_nuclei, v_electrons, nuclei.xy, electrons.xy
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
        return nuclei, electrons

    def update_particles(nuclei: Particle, electrons: Particle) -> tuple[Particle, Particle]:
        return jax.lax.scan(
            lambda args, _: (update_particles_once(*args), None),
            (nuclei, electrons),
            None,
            length=num_updates,
        )[0]

    return jax.jit(update_particles)


def run():
    dt = 0.00001
    pause = 0.01
    plot_frequency = 100
    num_steps = 1_000_000
    seed = 0

    nuclei, electrons = init_particles(
        jax.random.PRNGKey(seed),
        num_nuclei=4,
        num_electrons=32,
        nuclei_speed_scaling=18.0,
        electron_speed_scaling=8.0,
    )
    visualizer = Visualizer()
    update_particles = make_update_particles(dt, plot_frequency)
    print("Device:", nuclei.xy.devices())
    for step in range(num_steps // plot_frequency):
        t_0 = time.perf_counter()
        nuclei, electrons = jax.block_until_ready(update_particles(nuclei, electrons))
        fps = plot_frequency / (time.perf_counter() - t_0)
        visualizer.update_fig(nuclei, electrons, step * plot_frequency, fps, pause=pause)


if __name__ == "__main__":
    run()
