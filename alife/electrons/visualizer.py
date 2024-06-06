from functools import partial

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from alife.electrons.particles import Particle, P_CHARACHTERISTICS


class Visualizer:
    def __init__(
        self,
        min_grid_size: float = -1.0,
        max_grid_size: float = 1.0,
        wall: bool = False,
        wall_gap_size: float = 0.1,
    ):
        self.min_grid_size = min_grid_size
        self.max_grid_size = max_grid_size
        self.wall = wall
        self.wall_gap_size = wall_gap_size
        # Setup plot
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = plt.gca()
        self.colors = ["blue", "green"]
        self.cpu = jax.devices("cpu")[0]
        self.m_nuclei, self.m_electrons = P_CHARACHTERISTICS.mass
        self.radius_nuclei, self.radius_electrons = P_CHARACHTERISTICS.radius

    @partial(jax.jit, static_argnums=0)
    def total_kinetic_energy(self, nuclei: Particle, electrons: Particle) -> float:
        energy_nuclei = 0.5 * jnp.sum(self.m_nuclei * nuclei.alive * jnp.sum(nuclei.v**2, axis=-1))
        energy_electrons = 0.5 * jnp.sum(
            self.m_electrons * electrons.alive * jnp.sum(electrons.v**2, axis=-1)
        )
        return energy_nuclei + energy_electrons

    def update_fig(self, nuclei: Particle, electrons: Particle, step, fps, pause: float = 0.1):
        nuclei, electrons = jax.device_put((nuclei, electrons), self.cpu)
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

        for particles, radius, color in zip(
            [nuclei, electrons], [self.radius_nuclei, self.radius_electrons], self.colors
        ):
            for xy, alive in zip(particles.xy, particles.alive):
                if not alive:
                    continue
                circle = plt.Circle(xy, radius, color=color, fill=True, alpha=0.7)
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
            f"Total kinetic energy: {self.total_kinetic_energy(nuclei, electrons):.2e}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.title(f"Step {step}")
        plt.pause(pause)  # pause to update the plot
