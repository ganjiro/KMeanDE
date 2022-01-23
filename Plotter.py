import matplotlib.pyplot as plt


class Plotter():
    def __init__(self, X, Y, membership_vector, alpha=0.5):
        self.X = X
        self.Y = Y
        self.colors = membership_vector
        self.alpha = alpha

    def plot(self):
        plt.scatter(self.X, self.Y, c=self.colors, alpha=self.alpha)
        plt.show()
