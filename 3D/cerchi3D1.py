import Matrix
import Polytope
import trova_partizioni_vicine

from sklearn import datasets
import pandas as pd
import numpy as np
import json


dat1 = datasets.make_circles(n_samples=100,noise=0.05,random_state=500,factor=0.9)
X1 = dat1[0]
y1 = 2*np.ones(len(X1))
dat2 = datasets.make_circles(n_samples=200,noise=0.05,random_state=550,factor=0.5)
X2 = dat2[0]
y2 = dat2[1]
cerchio1=np.hstack((X1[:,0].reshape((len(X1),1)),X1[:,1].reshape((len(X1),1))-1,np.random.normal(0,0.05,len(X1)).reshape((len(X1),1))))
cerchio2=np.hstack((np.random.normal(0,0.05,len(X2)).reshape((len(X2),1)),X2[:,0].reshape((len(X2),1)),X2[:,1].reshape((len(X2),1))))
X = np.vstack([cerchio1,cerchio2])
y = np.hstack([y1,y2])






def SaveMondrianOutput(namefile,part,m):
	#part
	part.to_json(namefile+'_part.json')
	#m
	lista = list(np.array(m,dtype=object)[:,2])
	for i in lista:
		i.columns = i.columns.astype(str)
	with open(namefile+'_m.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in lista]))
	return


	
metric_list = ['variance','centroid_diff','centroid_ratio','min','min_corr']
exp = 50
t0 = 0
lifetime = 15
name = 'cerchi3D_'
dist_matrix = Matrix.DistanceMatrix(X)
dist_matrix.to_csv(name+'dist_matrix.txt',sep='\t',index=False)
#dist_matrix = pd.read_csv(name+'dist_matrix.txt',sep='\t')
number_of_iterations = 20






for i in range(len(metric_list)):
	for k in range(number_of_iterations):
	
		
		metric = metric_list[i]
		m_i,part_i = Polytope.Mondrian(X,t0,lifetime,dist_matrix,metric,exp)
		
		namefile = name+metric+'_'+str(k+1)
		SaveMondrianOutput(namefile,part_i,m_i)

		part = json.load(open(namefile+'_part.json','r'))
		part = pd.DataFrame(part)
		m = json.load(open(namefile+'_m.json','r'))


		list_m_leaf,list_p = trova_partizioni_vicine.Classification_BU(m,part,metric)
		list_p.reverse()
		list_m_leaf.reverse()
		
		with open(namefile+'_p.json', 'w') as f:
			f.write(json.dumps([df.to_dict() for df in list_p]))
		with open(namefile+'_m_leaf.json', 'w') as f:
			f.write(json.dumps([df for df in list_m_leaf]))
	
