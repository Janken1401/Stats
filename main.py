import numpy as np
from matplotlib import pyplot as plt

from stats import Stats, JointStats
N = 10000
std = 1
mu = 5
n_pdf = 50
std_2 = 2
mu_2 = 5
X = np.random.normal(mu, std, N)
Y = np.random.normal(mu_2, std_2, N)

j_stats = JointStats(X, Y, n_pdf)
print(np.corrcoef(X, Y))

x_pdf = j_stats.stats_x.compute_pdf()
plt.plot(j_stats.stats_x.levels, x_pdf)
plt.show()
plt.contour(j_stats.stats_x.levels - mu, j_stats.stats_y.levels - mu_2, j_stats.compute_joint_pdf())
plt.colorbar()
plt.show()