from functools import partial

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from alife.genome.particles import Particle, P_CHARACHTERISTICS


class Visualizer:
    def __init__(
        self,
        min_grid_size: float = -1.0,
        max_grid_size: float = 1.0,
        wall: bool = False,
        wall_gap_size: float = 0.1,
        energy_source_bool: bool = False,
        energy_source_size: bool = 0.5,
    ):
        self.min_grid_size = min_grid_size
        self.max_grid_size = max_grid_size
        self.wall = wall
        self.wall_gap_size = wall_gap_size
        self.energy_source_bool = energy_source_bool
        self.energy_source_size = energy_source_size
        # Setup plot
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = plt.gca()
        self.cpu = jax.devices("cpu")[0]

    @partial(jax.jit, static_argnums=0)
    def total_kinetic_energy(self, particles: Particle) -> float:
        energy = 0.5 * jnp.sum(P_CHARACHTERISTICS.mass * particles.alive * jnp.sum(particles.v**2, axis=-1))
        return energy

    def update_fig(self, particles: Particle, energy_source, step, fps, pause: float = 0.1):
        particles, energy_source = jax.device_put((particles, energy_source), self.cpu)
        self.ax.clear()
        plt.axis("off")
        plt.xlim(self.min_grid_size, self.max_grid_size)
        plt.ylim(self.min_grid_size, self.max_grid_size)
        plt.plot(
            [
                self.min_grid_size,
                self.max_grid_size,
                self.max_grid_size,
                self.min_grid_size,
                self.min_grid_size,
            ],
            [
                self.min_grid_size,
                self.min_grid_size,
                self.max_grid_size,
                self.max_grid_size,
                self.min_grid_size,
            ],
            color="black",
        )
        if self.wall:
            # plot horizontal wall at y = 0 with a small gap
            plt.plot(
                [self.min_grid_size, -self.wall_gap_size / 2],
                [0, 0],
                color="black",
            )
            plt.plot(
                [self.wall_gap_size / 2, self.max_grid_size],
                [0, 0],
                color="black",
            )
        if self.energy_source_bool:
            nrg_y, _ = energy_source
            rectangle = plt.Rectangle(
                (self.min_grid_size, nrg_y),
                self.energy_source_size,
                self.energy_source_size,
                color="red",
                alpha=0.2,
            )
            plt.gca().add_patch(rectangle)

        for xy, alive in zip(particles.xy, particles.alive):
            if not alive:
                continue
            circle = plt.Circle(xy, P_CHARACHTERISTICS.radius, color="blue", fill=True, alpha=0.7)
            plt.gca().add_patch(circle)
        plt.text(
            0.86 * self.max_grid_size,
            1.03 * self.max_grid_size,
            f"FPS: {fps:.2e}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.text(
            0.68 * self.max_grid_size,
            1.08 * self.max_grid_size,
            f"Total kinetic energy: {self.total_kinetic_energy(particles):.2e}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.title(f"Step {step}")
        plt.pause(pause)  # pause to update the plot
