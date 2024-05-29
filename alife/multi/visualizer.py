import matplotlib.pyplot as plt

from alife.multi.particles import Particle, P_CHARACHTERISTICS


class Visualizer:
    def __init__(self):
        # Setup plot
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = plt.gca()
        self.colors = ["blue", "green", "red"]

    def update_fig(self, particles: Particle, step, fps, pause: float = 0.1):
        self.ax.clear()
        plt.axis("off")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color="black")
        for xy, type, alive in zip(particles.xy, particles.type, particles.alive):
            if not alive:
                continue
            circle = plt.Circle(
                xy, P_CHARACHTERISTICS.radius[type], color=self.colors[type], fill=True, alpha=0.7
            )
            plt.gca().add_patch(circle)
        plt.text(
            0.9,
            1.03,
            f"FPS: {fps:.0f}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.title(f"Step {step}")
        plt.pause(pause)  # pause to update the plot
