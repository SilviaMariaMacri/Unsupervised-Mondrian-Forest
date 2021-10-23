import FileFinale
from sklearn import datasets
import numpy as np

import sys
makemoons3D,lifetime,exp,metric,number_of_iterations= sys.argv
#metric = ['variance','centroid_diff','centroid_ratio','min','min_corr']


name = 'moons3D_lambda'+lifetime+'_exp'+exp+'_'
t0 = 0

dat = datasets.make_moons(n_samples=80,noise=0.01,random_state=500)
X = dat[0]
y = dat[1]
np.random.seed(500)
dim3 = np.random.normal(0, 0.01, len(X))#np.zeros(len(Xdat))#
X = np.hstack((X[:,0].reshape((len(X),1)),X[:,1].reshape((len(X),1)),dim3.reshape((len(X),1))))


for i in range(6):
	dat = datasets.make_moons(n_samples=80,noise=0.01,random_state=i)
	Xdat = dat[0]
	ydat = dat[1]
	np.random.seed(i)
	dim3 = np.random.normal(0, 0.01, len(Xdat)) #np.zeros(len(Xdat))#
	Xdat = np.hstack((Xdat[:,0].reshape((len(Xdat),1)),Xdat[:,1].reshape((len(Xdat),1)),dim3.reshape((len(Xdat),1))))
	Xdat = Xdat +i*np.array([0.1,0.1,0.01])
	X = np.vstack([X,Xdat])
	y = np.hstack([y,ydat])



FileFinale.FunzioneFinale(name,
						  X,
						  t0,
						  int(lifetime),
						  int(exp),
						  metric,
						  int(number_of_iterations)
						  )