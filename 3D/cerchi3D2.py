
import FileFinale
from sklearn import datasets
import numpy as np

import sys
cerchi3D2,lifetime,exp,metric,number_of_iterations= sys.argv
#metric = ['variance','centroid_diff','centroid_ratio','min','min_corr']



dat1 = datasets.make_circles(n_samples=100,noise=0.05,random_state=500,factor=0.9)
X1 = dat1[0]
y1 = 2*np.ones(len(X1))
dat2 = datasets.make_circles(n_samples=200,noise=0.05,random_state=550,factor=0.5)
X2 = dat2[0]
y2 = dat2[1]
cerchio1=np.hstack((X1[:,0].reshape((len(X1),1)),
					X1[:,1].reshape((len(X1),1))-1,
					np.random.normal(0,0.05,len(X1)).reshape((len(X1),1))))
cerchio2=np.hstack((np.random.normal(0,0.05,len(X2)).reshape((len(X2),1)),
					X2[:,0].reshape((len(X2),1)),
					X2[:,1].reshape((len(X2),1))))
X = np.vstack([cerchio1,cerchio2])
y = np.hstack([y1,y2])


name = 'cerchi3D2_lambda'+lifetime+'_exp'+exp+'_'
t0 = 0



FileFinale.FunzioneFinale(name,
						  X,
						  t0,
						  int(lifetime),
						  int(exp),
						  metric,
						  int(number_of_iterations)
						  )