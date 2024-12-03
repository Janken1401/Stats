import numpy as np

class Stats:
    def __init__(self, X, n_pdf):
        self.n_pdf = n_pdf
        self.values = X
        self.dx = (np.max(self.values) - np.min(self.values)) / self.n_pdf
        self.levels = self.compute_levels()

        # self.compute_pdf()

    def run_pdf(self):
        self.mean_X = self.compute_mean()
        self.std_X = self.compute_std()
        self.var_X = self.compute_variance()
        self.skewness = self.compute_skewness()
        self.kurtosis = self.compute_kurtosis()

    def compute_levels(self):
        x_min = np.min(self.values)
        return np.array([np.mean((x_min + i * self.dx, x_min + (i + 1) * self.dx))
                             for i in range(self.n_pdf)])
    def compute_pdf(self):
        x_min = np.min(self.values)
        pdf_X = np.zeros(self.n_pdf)
        for x in self.values:
            k = int((x - x_min) // self.dx)
            k = min(k, self.n_pdf - 1)
            pdf_X[k] += 1
        somme = np.sum(pdf_X) * self.dx
        pdf_X /= somme

        return pdf_X

    def compute_mean(self):
        return np.sum(self.levels * self.pdfx * self.dx)

    def compute_variance(self):
        return np.sum((self.levels - self.mean_X) ** 2 * self.pdfx * self.dx)

    def compute_std(self):
        return np.sqrt(self.compute_variance())

    def compute_skewness(self):
        return np.sum((self.levels - self.mean_X) ** 3 * self.pdfx * self.dx) / self.std_X ** 3 / 2

    def compute_kurtosis(self):
        return np.sum((self.levels - self.mean_X) ** 4 * self.pdfx * self.dx) / self.std_X ** 4

    def compute_cdf(self):
        return np.cumsum(self.pdfx)

    def refind_pdf(self):
        df_dx = np.zeros(self.n_pdf)
        cdf = self.compute_cdf()
        for i in range(1, self.n_pdf - 1):
            df_dx[i] = (cdf[i + 1] - cdf[i]) / self.dx
        return df_dx


class JointStats:
    def __init__(self, X, Y, n_pdf):
        self.stats_x = Stats(X, n_pdf)
        self.stats_y = Stats(Y, n_pdf)
        self.n_pdf = n_pdf

    def compute_joint_cdf(self):
        joint_cdf = np.zeros((self.n_pdf, self.n_pdf))
        for i, x in enumerate(self.stats_x.levels):
            # proba_x = np.where(self.stats_x.values < x)

            for j, y in enumerate(self.stats_y.levels):
                # proba_y = np.where(self.stats_y.values < y)
                # proba_and = np.intersect1d(proba_x, proba_y)
                joint_cdf[i, j] = np.sum((self.stats_x.values < x) & (self.stats_y.values < y))

        return joint_cdf

    def compute_joint_pdf(self):
        dx = self.stats_x.dx
        dy = self.stats_y.dx

        joint_cdf = self.compute_joint_cdf()
        pdf_xy = np.zeros((self.n_pdf, self.n_pdf))
        for i in range(1, self.n_pdf - 1):
            for j in range(1, self.n_pdf - 1):
                pdf_xy[i, j] = (joint_cdf[i + 1, j + 1]
                         - joint_cdf[i, j + 1]
                         - joint_cdf[i + 1, j]
                         + joint_cdf[i, j]) / (dx * dy)

        return pdf_xy / (np.sum(pdf_xy) * dx * dy)