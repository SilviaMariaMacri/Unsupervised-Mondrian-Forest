import FileFinale
from sklearn import datasets
import numpy as np

import sys
sfera,lifetime,exp,metric,number_of_iterations= sys.argv
#metric = ['variance','centroid_diff','centroid_ratio','min','min_corr']


name = 'sfera_lambda'+lifetime+'_exp'+exp+'_'
t0 = 0


rho = 0.25
np.random.seed(0)
theta = np.random.uniform(low=0, high=2*np.pi, size=120)
np.random.seed(1)
phi = np.random.uniform(low=0, high=np.pi, size=120)


np.random.seed(3)
eps1 = np.random.normal(0,0.01,120)
np.random.seed(4)
eps2 = np.random.normal(0,0.01,120)
np.random.seed(5)
eps3 = np.random.normal(0,0.01,120)


x = rho*np.sin(phi)*np.cos(theta) + eps1
y = rho*np.sin(phi)*np.sin(theta) + eps2
z = rho*np.cos(phi) + eps3



Xsph = np.vstack([x,y,z]).T
ysph = np.zeros(len(Xsph))




blob = datasets.make_blobs(n_samples=40,n_features=3,centers=[(0,0,0)],cluster_std=0.02,random_state=18)
Xb = blob[0]
yb = np.ones(len(Xb))

X = np.vstack([Xsph,Xb])
y = np.hstack([ysph,yb])



FileFinale.FunzioneFinale(name,
						  X,
						  t0,
						  int(lifetime),
						  int(exp),
						  metric,
						  int(number_of_iterations)
						  )