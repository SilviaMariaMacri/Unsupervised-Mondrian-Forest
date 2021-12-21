import FileFinale
import numpy as np
from sklearn import datasets


import sys
cilindro,lifetime,exp,metric,number_of_iterations= sys.argv
#metric = ['variance','centroid_diff','centroid_ratio','min','min_corr']


name = 'cilindro_nuovo2_lambda'+lifetime+'_exp'+exp+'_'
t0 = 0


np.random.seed(0)
angle = np.random.uniform(low=0, high=2*np.pi, size=200)
z = np.random.uniform(low=0, high=0.5, size=200)

np.random.seed(2)
x = 0.5*np.cos(angle) + np.random.normal(0,0.05,200)
np.random.seed(3)
y = 0.5*np.sin(angle) + np.random.normal(0,0.05,200)

X_cilindro = np.vstack([x,y,z])
X_cilindro = X_cilindro.T

y_cilindro = 2*np.ones(len(X_cilindro))


np.random.seed(1)
x1 = np.random.uniform(low=-0.1, high=0.1, size=50)
y1 = np.random.uniform(low=-0.1, high=0.1, size=50)
z1 = np.random.uniform(low=0, high=0.5, size=50)

X1 = np.vstack([x1,y1,z1])
X1 = X1.T
Y1 = np.ones(len(X1))


dat = datasets.make_blobs(n_samples=[50], n_features=3, cluster_std=0.1,center_box=(0.8,0.8,0.8), random_state=0, return_centers=True)
X2 = dat[0]
Y2 = dat[1]



X = np.vstack([X_cilindro,X1,X2])
y = np.hstack([y_cilindro,Y1,Y2])




FileFinale.FunzioneFinale(name,
						  X,
						  t0,
						  float(lifetime),
						  float(exp),
						  metric,
						  int(number_of_iterations)
						  )