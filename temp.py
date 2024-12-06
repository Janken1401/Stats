import numpy as np
from plot import AFFICHAGE


N = 10000
X = np.random.normal(5, 0.5, N)
Y = np.random.normal(5, 0.5, N)

param = dict(N=10000,
             npdf=250,
             X=X,
             Y=Y)

r = AFFICHAGE(TP='TP3', question='all', param=param)
