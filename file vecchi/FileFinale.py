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
	lista = list(np.array(m,dtype=object)[:,2])
	for i in lista:
		i.columns = i.columns.astype(str)
	with open(namefile+'_m.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in lista]))
	return
	


def FunzioneFinale(name,X,t0,lifetime,exp,metric,number_of_iterations):
	
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
		
		
		#ricalcolo vicini
		part = part.drop('neighbors',axis=1)
	
		lista_complessiva = []
		part_leaf_true = part.query('leaf==True').copy()
		part_leaf_true.index = np.arange(len(part_leaf_true))
		for i in range(len(part_leaf_true)):
			lista_partizione_singola = []
			for j in range(len(part_leaf_true)):
				poly_i = pc.Polytope(np.array(part_leaf_true['polytope'].iloc[i]['A']),np.array(part_leaf_true['polytope'].iloc[i]['b']))
				poly_j = pc.Polytope(np.array(part_leaf_true['polytope'].iloc[j]['A']),np.array(part_leaf_true['polytope'].iloc[j]['b']))
				if (pc.is_adjacent(poly_i,poly_j) == True) and (part_leaf_true['part_number'].iloc[i]!=part_leaf_true['part_number'].iloc[j]):
					lista_partizione_singola.append(int(part_leaf_true['part_number'].iloc[j]))
			lista_complessiva.append(lista_partizione_singola)
		
		part_leaf_true['neighbors'] = lista_complessiva			
		part = pd.merge(part,part_leaf_true[['part_number','neighbors']],how='left',right_on='part_number',left_on='part_number')		
		# fine ricalcolo vicini 		
	
			
		list_m_leaf,list_p = Merging.Classification_BU(m,part,metric)
		list_p.reverse()
		list_m_leaf.reverse()
			
		with open(namefile+'_p.json', 'w') as f:
			f.write(json.dumps([df.to_dict() for df in list_p]))
		with open(namefile+'_m_leaf.json', 'w') as f:
			f.write(json.dumps([df for df in list_m_leaf]))
		
	return 