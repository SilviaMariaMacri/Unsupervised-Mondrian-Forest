# -*- coding: utf-8 -*-
import FileFinale
from sklearn import datasets

import sys
circles2D,lifetime,exp,metric,number_of_iterations= sys.argv


name = 'circles2D_exp'+exp+'_'
dat = datasets.make_circles(n_samples=200,noise=0.05,random_state=500,factor=0.5)

X = dat[0]
t0 = 0


FileFinale.FunzioneFinale(name,
						  X,
						  t0,
						  int(lifetime),
						  int(exp),
						  metric,
						  int(number_of_iterations)
						  )