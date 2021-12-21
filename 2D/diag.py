# -*- coding: utf-8 -*-
import FileFinale
import numpy as np

import sys
diag,lifetime,exp,metric,number_of_iterations= sys.argv



name = 'diag_exp'+exp+'_'

a = 50
b1 = 0.07
b2 = 0.06
mean1 = (0.2, 0)
cov1 = [[b1,b2], [b2,b1]]
np.random.seed(2)
x1 = np.random.multivariate_normal(mean1, cov1, a)
mean2 = (0.6, -0.5) 
cov2 = [[b1,b2], [b2,b1]]
np.random.seed(1)
x2 = np.random.multivariate_normal(mean2, cov2, a)
X = np.vstack([x1,x2])

t0 = 0


FileFinale.FunzioneFinale(name,
						  X,
						  t0,
						  int(lifetime),
						  int(exp),
						  metric,
						  int(number_of_iterations)
						  )