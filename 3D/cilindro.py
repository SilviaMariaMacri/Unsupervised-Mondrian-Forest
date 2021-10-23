import FileFinale
import numpy as np
from sklearn import datasets

import sys
cilindro,lifetime,exp,metric,number_of_iterations= sys.argv
#metric = ['variance','centroid_diff','centroid_ratio','min','min_corr']


name = 'cilindro_lambda'+lifetime+'_exp'+exp+'_'
t0 = 0



from scipy.spatial.transform import Rotation as R
rotation_axis = np.array([0, 0, 1])
angle = np.linspace(0,360,20)
Xinit=[2,2,0]
X_cilindro = []


lunghezza_verticale = 5
for j in range(lunghezza_verticale):
	for i in range(len(angle)-1):
		rotation_degrees = angle[i]
		rotation_radians = np.radians(rotation_degrees)
		rotation_vector = rotation_radians * rotation_axis
		rotation = R.from_rotvec(rotation_vector)
		X_rot = rotation.apply(Xinit) + np.random.normal(0,0.05,len(Xinit)) +[j*0.7,j*0.7,j*0.1]
		X_cilindro.append(list(X_rot))

X_cilindro = np.array(X_cilindro) 
y_cilindro = np.zeros(len(X_cilindro))


X1 = []
mean = [0, 0, 0]
cov = [[0.01, 0,0], [0,0.01,0],[0,0,0.001]]
for j in range(lunghezza_verticale*10):
	np.random.seed(j)
	X1_i =  [j*0.07,j*0.07,j*0.01] + np.random.multivariate_normal(mean,cov)
	X1.append(X1_i)
y1 = np.ones(len(X1))	


dat = datasets.make_blobs(n_samples=[50],centers=[[3,3,-0.5]],n_features=3,cluster_std=[[0.5,0.5,0.05]],random_state=50)
X2=dat[0]
y2 = 2*np.ones(len(X2))


X = np.vstack([X_cilindro,X1,X2])
y = np.hstack([y_cilindro,y1,y2])



FileFinale.FunzioneFinale(name,
						  X,
						  t0,
						  float(lifetime),
						  float(exp),
						  metric,
						  int(number_of_iterations)
						  )