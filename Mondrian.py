import Matrix
import Partitioning
import Merging

import pandas as pd
import numpy as np
import json

import polytope as pc 


def SaveMondrianOutput(namefile,part,m):
	#part
	part.to_json(namefile+'_part.json')
	#m
	lista = list(np.array(m,dtype=object)[:,3])
	for i in lista:
		i.columns = i.columns.astype(str)
	with open(namefile+'_m.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in lista]))
	return
	


def MondrianTree(name,X,t0,lifetime,exp,metric,number_of_iterations):
	
	dist_matrix = Matrix.DistanceMatrix(X)
	#dist_matrix.to_csv(name+'dist_matrix.txt',sep='\t',index=False)
	#dist_matrix = pd.read_csv(name+'dist_matrix.txt',sep='\t')
	
	for k in range(number_of_iterations):
		
		#metric = metric_list[i]
		m_i,part_i = Partitioning.Mondrian(X,t0,lifetime,dist_matrix,metric,exp)
			
		namefile = name+metric+'_'+str(k+1)
		SaveMondrianOutput(namefile,part_i,m_i)
	
		part = json.load(open(namefile+'_part.json','r'))
		part = pd.DataFrame(part)
		m = json.load(open(namefile+'_m.json','r'))
		
		
		#calcolo vicini
		neighbors_list = []
		leaves = part.query('leaf==True').copy()
		leaves.index = np.arange(len(leaves))
		for i in range(len(leaves)):
			poly_i = pc.Polytope(np.array(leaves['polytope'].iloc[i]['A']),np.array(leaves['polytope'].iloc[i]['b']))
			neighbors = []
			for j in range(len(leaves)):
				poly_j = pc.Polytope(np.array(leaves['polytope'].iloc[j]['A']),np.array(leaves['polytope'].iloc[j]['b']))
				if (pc.is_adjacent(poly_i,poly_j) == True) and (leaves['part_number'].iloc[i]!=leaves['part_number'].iloc[j]):
					neighbors.append(int(leaves['part_number'].iloc[j]))
			neighbors_list.append(neighbors)
		
		leaves['neighbors'] = neighbors_list			
		part = pd.merge(part,leaves[['part_number','neighbors']],how='left',right_on='part_number',left_on='part_number')		
		# fine ricalcolo vicini 		


		list_m_leaf,list_p = Merging.Classification_BU(m,part,metric)
		list_p.reverse()
		list_m_leaf.reverse()
			
		with open(namefile+'_p.json', 'w') as f:
			f.write(json.dumps([df.to_dict() for df in list_p]))
		with open(namefile+'_m_leaf.json', 'w') as f:
			f.write(json.dumps([df for df in list_m_leaf]))
		
	return 