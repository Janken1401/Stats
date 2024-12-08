import numpy as np
from matplotlib import pyplot as plt


class Stats:
    def __init__(self, X, n_pdf):
        self.n_pdf = n_pdf
        self.values = X
        self.dx = (np.max(self.values) - np.min(self.values)) / self.n_pdf
        self.levels = self.levels()

    def levels(self):
        x_min = np.min(self.values)
        x_levels = np.zeros(self.n_pdf)
        for i in range(self.n_pdf):
            x_levels[i] = x_min + i * self.dx + self.dx / 2

        return x_levels

    def pdf(self):
        x_min = np.min(self.values)
        pdf_X = np.zeros(self.n_pdf)
        for x in self.values:
            k = int((x - x_min) // self.dx)
            k = min(k, self.n_pdf - 1)
            pdf_X[k] += 1
        somme = np.sum(pdf_X) * self.dx
        pdf_X /= somme

        return pdf_X

    def mean(self):
        return np.sum(self.levels * self.pdf() * self.dx)

    def variance(self):
        return np.sum((self.levels - self.mean()) ** 2 * self.pdf() * self.dx)

    def std(self):
        return np.sqrt(self.variance())

    def skewness(self):
        return np.sum((self.levels - self.mean()) ** 3 * self.pdf() * self.dx) / self.std() ** 3 / 2

    def kurtosis(self):
        return np.sum((self.levels - self.mean()) ** 4 * self.pdf() * self.dx) / self.std() ** 4

    def cdf(self):
        return np.cumsum(self.pdf()) * self.dx

    def refind_pdf(self):
        df_dx = np.zeros(self.n_pdf)
        cdf = self.cdf()
        for i in range(1, self.n_pdf):
            df_dx[i] = (cdf[i] - cdf[i - 1]) / self.dx
        return df_dx / (np.sum(df_dx) * self.dx)


class JointStats:
    def __init__(self, X, Y, n_pdf):
        self.stats_x = Stats(X, n_pdf)
        self.stats_y = Stats(Y, n_pdf)
        self.n_pdf = n_pdf
        self.joint_pdf = self._joint_pdf()

    def joint_cdf(self):
        joint_cdf = np.zeros((self.n_pdf, self.n_pdf))
        for i, x in enumerate(self.stats_x.levels):
            for j, y in enumerate(self.stats_y.levels):
                joint_cdf[i, j] = np.sum((self.stats_x.values < x) & (self.stats_y.values < y))

        return joint_cdf

    def _joint_pdf(self):
        dx = self.stats_x.dx
        dy = self.stats_y.dx
        joint_cdf = self.joint_cdf()
        pdf_xy = np.zeros((self.n_pdf, self.n_pdf))
        for i in range(self.n_pdf - 1):
            for j in range(self.n_pdf - 1):
                pdf_xy[i, j] = (joint_cdf[i + 1, j + 1]
                         - joint_cdf[i, j + 1]
                         - joint_cdf[i + 1, j]
                         + joint_cdf[i, j]) / (dx * dy)
        return pdf_xy / (np.sum(pdf_xy) * dx * dy)

    def mixt_moment(self, k, n):
        dx = self.stats_x.dx
        dy = self.stats_y.dx
        return np.sum(self.stats_x.values ** k
                      * self.stats_y.values ** n
                      * self.joint_pdf() * dx * dy)

    def mixt_centered_moment(self, k, n):
        dx = self.stats_x.dx
        dy = self.stats_y.dx
        x_mean = self.stats_x.mean()
        y_mean = self.stats_y.mean()

        somme = 0
        for i in range(self.n_pdf):
            for j in range(self.n_pdf):
                somme += ((self.stats_x.levels[i] - x_mean) ** k
                        *(self.stats_y.levels[j] - y_mean) ** n
                          * self.joint_pdf[i, j]) * dx * dy
        return somme

    def covariance(self):
        return self.mixt_centered_moment(1, 1)

    def correlation(self):
        std_x = self.stats_x.std()
        std_y = self.stats_y.std()
        return self.covariance() / (std_x * std_y)

    def show_pdf(self):
        X, Y = np.meshgrid(self.stats_x.levels, self.stats_y.levels)
        fig, ax = plt.subplots()
        surf = ax.contourf(X - self.stats_x.mean(), Y - self.stats_y.mean(), self.joint_pdf,
                               antialiased=False, linewidth=0)
        plt.colorbar(surf)
        plt.show()

    def scatter_plot(self):
        fig, ax = plt.subplots()
        ax.scatter(self.stats_x.values, self.stats_y.values, s=0.5)
        ax.set_xlabel('X - value')
        ax.set_ylabel('Y - value')
        ax.set_title('Scatter plot')
        plt.show()

