import FileFinale
import numpy as np
from sklearn import datasets

import sys
cilindro,lifetime,exp,metric,number_of_iterations= sys.argv
#metric = ['variance','centroid_diff','centroid_ratio','min','min_corr']


name = 'cilindro_vecchio_lambda'+lifetime+'_exp'+exp+'_'
t0 = 0



from scipy.spatial.transform import Rotation as R
rotation_axis = np.array([0, 0, 1])
angle = np.linspace(0,360,20)
Xinit=[2,2,0]
X = []
Xb = [0,0,0]
for i in range(len(angle)-1):
	rotation_degrees = angle[i]
	rotation_radians = np.radians(rotation_degrees)
	rotation_vector = rotation_radians * rotation_axis
	rotation = R.from_rotvec(rotation_vector)
	X_rot = rotation.apply(Xinit)
	X.append(list(X_rot))
Xinit = np.array(X) 
y=np.zeros(len(Xinit))
Xbtt = [0,0,0]
for i in range(3):
	np.random.seed(0)
	Xt = np.array(Xinit) + np.random.normal(0,0.1,(len(Xinit),3)) +i*0.3
	yt = np.zeros(len(Xt))
	X = np.vstack([X,Xt])
	np.random.seed(1)
	Xbt = np.array(Xb) + np.random.normal(0,0.4,(10,3)) +i*0.3
	Xbtt = np.vstack([Xbtt,Xbt])
	
y1 = np.zeros(len(X))
y2 = np.ones(len(Xbtt))
X = np.vstack([X,Xbtt])
y = np.hstack([y1,y2])



FileFinale.FunzioneFinale(name,
						  X,
						  t0,
						  float(lifetime),
						  float(exp),
						  metric,
						  int(number_of_iterations)
						  )