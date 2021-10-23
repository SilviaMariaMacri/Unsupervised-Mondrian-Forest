
No scherzavo non questo




import Matrix
import Polytope
import trova_partizioni_vicine

from sklearn import datasets
import pandas as pd
import numpy as np
import json
'''
#from scipy.spatial.transform import Rotation as R
dat = datasets.make_moons(n_samples=100,noise=0.05,random_state=500)
X = dat[0]
y = dat[1]
np.random.seed(500)
dim3 = np.random.normal(0, 0.01, len(X))#np.zeros(len(Xdat))#
X = np.hstack((X[:,0].reshape((len(X),1)),X[:,1].reshape((len(X),1)),dim3.reshape((len(X),1))))


for i in range(20):
	dat = datasets.make_moons(n_samples=100,noise=0.05,random_state=i)
	Xdat = dat[0]
	ydat = dat[1]
	np.random.seed(i)
	dim3 = np.random.normal(0, 0.01, len(Xdat)) #np.zeros(len(Xdat))#
	Xdat = np.hstack((Xdat[:,0].reshape((len(Xdat),1)),Xdat[:,1].reshape((len(Xdat),1)),dim3.reshape((len(Xdat),1))))
	Xdat = Xdat +i*np.array([0.05,0.05,0.01])
	X = np.vstack([X,Xdat])
	y = np.hstack([y,ydat])
'''



dat = datasets.make_circles(n_samples=100,noise=0.05,random_state=500,factor=0.5)
#dat = datasets.make_moons(n_samples=20,noise=0.08,random_state=500)
X = dat[0]




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
lifetime = 6
name = 'circles_'
dist_matrix = Matrix.DistanceMatrix(X)
dist_matrix.to_csv(name+'dist_matrix.txt',sep='\t',index=False)
#dist_matrix = pd.read_csv(name+'dist_matrix.txt',sep='\t')
number_of_iterations = 20


for i in range(len(metric_list)):
	for k in range(number_of_iterations):
	
		
		metric = metric_list[i]
		m_i,part_i = Polytope.Mondrian(X,t0,lifetime,dist_matrix,metric,exp)
		
		namefile = name+metric#+'_'+str(i+1)
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
	
