import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from stats import Stats, JointStats

N = 100000
n_pdf = 100

mu_X = 5
std_X = 1

mu_Y = 2
std_Y = 0.5

std_Xr = 1
mu_Xr = 0

std_Yr = 1
mu_Yr = 0

# X, Y = np.random.normal(mu_X, std_X, N), np.random.normal(mu_Y, std_Y, N)


Xr, Yr = np.random.normal(mu_Xr, std_Xr, N), np.random.normal(mu_Yr, std_Yr, N)

XYr_stats = JointStats(Xr, Yr, n_pdf)
corr_xy = XYr_stats.correlation()


X = mu_Xr + std_Xr * Xr
Y_star = corr_xy * Xr + np.sqrt((1 - corr_xy ** 2 )) * Yr
Y = mu_Yr + std_Yr * Y_star

j_stats = JointStats(X, Y, n_pdf)
X_stats = Stats(X, n_pdf)
Y_stats = Stats(Y, n_pdf)

X, Y = np.meshgrid(j_stats.stats_x.levels, j_stats.stats_y.levels)

fig, ax = plt.subplots()
ax.plot_surface(X - mu_X, Y - mu_Y, j_stats.joint_pdf(),
                       antialiased=False, linewidth=0,
                       cmap=cm.inferno)
fig.colorbar()
plt.show()

fig, ax = plt.subplots()

XY_pdf = np.outer(X_stats.pdf(), Y_stats.pdf())
surf_2 = ax.plot_surface(X - mu_X, Y - mu_Y, XY_pdf,
                       antialiased=False, linewidth=0,
                       cmap=cm.inferno)
fig.colorbar()

plt.show()







# x_pdf = j_stats.stats_x.pdf()
# plt.plot(j_stats.stats_x.levels, x_pdf)
# plt.show()
# X, Y = np.meshgrid(j_stats.stats_x.levels, j_stats.stats_y.levels)
#
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(X- mu, Y- mu_2, j_stats.joint_pdf(),
#                        antialiased=False, linewidth=0,
#                        cmap=cm.inferno)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()

