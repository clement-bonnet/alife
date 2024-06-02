from functools import partial

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from alife.electrons.particles import Particle, P_CHARACHTERISTICS


class Visualizer:
    def __init__(self):
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
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color="black")
        for particles, radius, color in zip(
            [nuclei, electrons], [self.radius_nuclei, self.radius_electrons], self.colors
        ):
            for xy, alive in zip(particles.xy, particles.alive):
                if not alive:
                    continue
                circle = plt.Circle(xy, radius, color=color, fill=True, alpha=0.7)
                plt.gca().add_patch(circle)
        plt.text(
            0.86,
            1.03,
            f"FPS: {fps:.2e}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.text(
            0.68,
            1.08,
            f"Total kinetic energy: {self.total_kinetic_energy(nuclei, electrons):.2e}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.title(f"Step {step}")
        plt.pause(pause)  # pause to update the plot
