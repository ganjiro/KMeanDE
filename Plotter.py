import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
# np.random.seed(19680801)
#
# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = (30 * np.random.rand(N)) ** 2  # 0 to 15 point radii
#
# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()


class Plotter():
    def __init__(self, X, Y, membership_vector, alpha=0.5):
        self.X = X
        self.Y = Y
        self.colors = membership_vector
        self.alpha = alpha

    def plot(self):
        plt.scatter(self.X, self.Y, c=self.colors, alpha=self.alpha)
        plt.show()
