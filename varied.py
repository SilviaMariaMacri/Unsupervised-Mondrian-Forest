import FileFinale
from sklearn import datasets

import sys
varied,lifetime,exp,metric,number_of_iterations= sys.argv

name = 'varied_exp'+exp+'_'
varied = datasets.make_blobs(n_samples=200,n_features=3,cluster_std=[1.0, 2.5, 0.5],random_state=150)
X=varied[0]
t0 = 0


FileFinale.FunzioneFinale(name,
						  X,
						  t0,
						  int(lifetime),
						  int(exp),
						  metric,
						  int(number_of_iterations)
						  )