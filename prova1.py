'''
import Matrix
import Polytope
import trova_partizioni_vicine'''
from sklearn import datasets
import pandas as pd
import numpy as np
import json


dat = datasets.make_circles(n_samples=20,noise=0.05,random_state=500,factor=0.5)
X = dat[0]
print(X)
df =pd.DataFrame(X)
df.to_csv('df.txt',sep=',')


'''
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
	


# Polytope dimensione generica + classificazione 

t0 = 0
lifetime = 1.5
dist_matrix = Matrix.DistanceMatrix(X)
number_of_iterations = 1
name = 'prova_'
for i in range(number_of_iterations):

	#i=1
	m_i,part_i = Polytope.Mondrian(X,t0,lifetime,dist_matrix)
	namefile = name+str(i+1)
	SaveMondrianOutput(namefile,part_i,m_i)

	part = json.load(open(namefile+'_part.json','r'))
	part = pd.DataFrame(part)
	m = json.load(open(namefile+'_m.json','r'))


	#tagli_paralleli = False #True#,False
	score = 'min' #'var','centroid'
	weight = 'diff_min' #'var_ratio','ratio_centroid','diff_centroid',
	#Classification(part,m,X,namefile,score,weight,tagli_paralleli)
	#Classification_BU(m,part,weight,score,namefile)
	list_m_leaf,list_p = trova_partizioni_vicine.Classification_BU(m,part,weight)
	list_p.reverse()
	list_m_leaf.reverse()
	
	with open(namefile+'_p.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in list_p]))
	with open(namefile+'_m_leaf.json', 'w') as f:
		f.write(json.dumps([df for df in list_m_leaf]))
		
print('fatto')
'''