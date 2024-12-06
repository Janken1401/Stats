import numpy as np
from plot import AFFICHAGE


N = 100000
npdf = 50

mu_X = 5
std_X = 1

mu_Y = 2
std_Y = 0.5

std_Xr = 1
mu_Xr = 0

std_Yr = 1
mu_Yr = 0

corr_xy = 0.7


Xr, Yr = np.random.normal(mu_Xr, std_Xr, N), np.random.normal(mu_Yr, std_Yr, N)
Y_star = corr_xy * Xr + np.sqrt((1 - corr_xy ** 2 )) * Yr

X1, Y1 = np.random.normal(mu_X, std_X, N), np.random.normal(mu_Y, std_Y, N)
X2, Y2 = mu_Xr + std_Xr * Xr, mu_Yr + std_Yr * Y_star


param = dict(N=N,
             npdf=npdf,
             X=X1,
             Y=Y1)

r = AFFICHAGE(TP='TP3', question='all', param=param)
