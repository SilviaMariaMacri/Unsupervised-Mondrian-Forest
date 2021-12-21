# -*- coding: utf-8 -*-
import FileFinale
from sklearn import datasets

import sys
moons2D,lifetime,exp,metric,number_of_iterations= sys.argv
#metric = ['variance','centroid_diff','centroid_ratio','min','min_corr']


name = 'moons2D_exp'+exp+'_'
dat = datasets.make_moons(n_samples=150,noise=0.05,random_state=500)
X = dat[0]
t0 = 0



FileFinale.FunzioneFinale(name,
						  X,
						  t0,
						  float(lifetime),
						  float(exp),
						  metric,
						  int(number_of_iterations)
						  )