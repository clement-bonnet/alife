import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, num_particles: int):
        self.num_particles = num_particles
        # Setup plot
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = plt.gca()

    def update_fig(self, positions, radii, step, fps, pause: float = 0.1):
        self.ax.clear()
        plt.axis("off")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="black")
        for i in range(self.num_particles):
            circle = plt.Circle(positions[i], radii[i], color="blue", fill=True)
            plt.gca().add_patch(circle)
        plt.text(
            0.9,
            1.03,
            f"FPS: {fps:.0f}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.title(f"Step {step + 1}")
        plt.pause(pause)  # pause to update the plot
