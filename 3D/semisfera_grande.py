import FileFinale
from sklearn import datasets
import numpy as np

import sys
semisfera,lifetime,exp,metric,number_of_iterations= sys.argv
#metric = ['variance','centroid_diff','centroid_ratio','min','min_corr']


name = 'semisfera_lambda'+lifetime+'_exp'+exp+'_'
t0 = 0


from scipy.spatial.transform import Rotation as R
rotation_axis = np.array([0, 0, 1])
angle = np.linspace(0,170,30)
Xinit=[1,0,0]
X = []
Xb = [0,0,0]

for i in range(len(angle)):
	rotation_degrees = angle[i]
	rotation_radians = np.radians(rotation_degrees)
	rotation_vector = rotation_radians * rotation_axis
	rotation = R.from_rotvec(rotation_vector)
	X_rot = rotation.apply(Xinit)
	X.append(list(X_rot))
X = np.array(X) #+ np.random.normal(0,0.1,(len(X),3))
Xinit = X.copy()

for i in range(len(angle)):
	rotation_axis = np.array([1, 0, 0])
	rotation_degrees = angle[i]
	rotation_radians = np.radians(rotation_degrees)
	rotation_vector = rotation_radians * rotation_axis
	rotation = R.from_rotvec(rotation_vector)
	X_rot = rotation.apply(Xinit + np.random.normal(0,0.05,(len(Xinit),3)))
	X = np.vstack([X,X_rot])
y = np.zeros(len(X))

blob = datasets.make_blobs(n_samples=300,n_features=3,centers=[(0,0,0.4)],cluster_std=0.1,random_state=18)
Xb = blob[0]
yb = np.ones(len(Xb))

X = np.vstack([X,Xb])
y = np.hstack([y,yb])



FileFinale.FunzioneFinale(name,
						  X,
						  t0,
						  int(lifetime),
						  int(exp),
						  metric,
						  int(number_of_iterations)
						  )