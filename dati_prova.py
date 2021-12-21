# -*- coding: utf-8 -*-
import Mondrian
from sklearn import datasets

import sys
dati_prova,lifetime,exp,metric,number_of_iterations= sys.argv


name = 'dati_prova_lambda'+lifetime+'_exp'+exp+'_'
#dat = datasets.make_circles(n_samples=200,noise=0.05,random_state=500,factor=0.5)
dat = datasets.make_circles(n_samples=20,noise=0.05,random_state=500,factor=0.5)

X = dat[0]
y = dat[1]
t0 = 0


Mondrian.Mondrian(name,
						  X,
						  t0,
						  int(lifetime),
						  int(exp),
						  metric,
						  int(number_of_iterations)
						  )