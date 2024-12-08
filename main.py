import numpy as np
import pandas as pd
from jupyter_lsp.specs import julia
from matplotlib import pyplot as plt
from matplotlib import cm
from stats import Stats, JointStats
import seaborn as sns
N = 10000
n_pdf = 50

mu_X = 5
std_X = 1

mu_Y = 2
std_Y = 0.5

std_Xr = 1
mu_Xr = 0

std_Yr = 1
mu_Yr = 0


Xr, Yr = np.random.normal(mu_Xr, std_Xr, N), np.random.normal(mu_Yr, std_Yr, N)

XYr_stats = JointStats(Xr, Yr, n_pdf)
corr_xy = XYr_stats.correlation()


coeff_XY = 0.7
X = mu_Xr + std_Xr * Xr
Y_star = coeff_XY * Xr + np.sqrt((1 - coeff_XY ** 2 )) * Yr
Y = mu_Yr + std_Yr * Y_star


j_stats = JointStats(X, Y, n_pdf)
print(np.corrcoef(X, Y))
print(j_stats.correlation())
X_stats = Stats(X, n_pdf)
Y_stats = Stats(Y, n_pdf)

j_stats.show_pdf()
j_stats.scatter_plot()


