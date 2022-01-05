import numpy as np
import pandas as pd
#import networkx as nx
import copy
import polytope as pc 

from Metrics import variance_metric,centroid_metric,min_dist_metric




def neighbors(part):
		
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
	part_neigh = pd.merge(part,leaves[['part_number','neighbors']],how='left',right_on='part_number',left_on='part_number')		
	
	return part_neigh			




def merge_two_polytopes(m_leaf_init,p_init,part_to_remove,part_to_merge):
	
	p = p_init.copy(deep=True)
	m_leaf = copy.deepcopy(m_leaf_init)

	index1 = p[p['part_number']==part_to_remove].index[0]
	index2 = p[p['part_number']==part_to_merge].index[0]
	#unisco i dati in m_leaf
	data1 = pd.DataFrame(m_leaf[index1]).copy()
	
	data2 = pd.DataFrame(m_leaf[index2]).copy()
	data2 = pd.concat([data2,data1])
	data2.index = np.arange(len(data2))
	m_leaf[index2] = copy.deepcopy(data2.to_dict())
	
	neigh = copy.deepcopy(list(p['neighbors']))
	merged_part = copy.deepcopy(list(p['merged_part']))

	#sostituisco vicini in tutte le partizioni di p e aggiungo vicini di single_part a part unita
	neigh[index1].remove(part_to_merge)
	neigh[index2].remove(part_to_remove)
	for j in neigh[index1]:
		if j not in neigh[index2]:
			neigh[index2].append(j)
		neigh[p[p['part_number']==j].index[0]].remove(part_to_remove)
		if part_to_merge not in neigh[p[p['part_number']==j].index[0]]:
			neigh[p[p['part_number']==j].index[0]].append(float(part_to_merge))#float(part_to_merge)

	for j in merged_part[index1]:
		merged_part[index2].append(float(j))#float(j)
	merged_part[index2].append(part_to_remove)
	
	p['neighbors'] = neigh
	p['merged_part'] = merged_part
	
	p = p.drop(index1)
	p.index = np.arange(len(p))
	m_leaf = copy.deepcopy(np.delete(m_leaf, index1).tolist())
	
	return m_leaf,p









#unisce partizione con solo un dato a quella più vicina
def merge_single_data(m_leaf_init,p_init): 
	
	p = p_init.copy(deep=True)
	m_leaf = copy.deepcopy(m_leaf_init)#m.copy()#


	list_part = list(p['part_number']).copy()
	for i in list_part:
		index1 = p[p['part_number']==i].index[0].copy()
		data1 = pd.DataFrame(m_leaf[index1]).copy()
		#considero solo partizioni con unico dato all'interno
		if len(data1) > 1:
			continue
		data1 = data1.drop('index',axis=1) 
		data1 = np.array(data1)
		#cerco minima distanza con part vicine
		metric = []
		for j in p['neighbors'].iloc[index1]:
			data2 = pd.DataFrame(m_leaf[p[p['part_number']==j].index[0]]).copy()
			if len(data2) > 1:
				data2 = data2.drop('index',axis=1)
				data2 = np.array(data2)
				min_dist_between_subspaces,mean,min_dist1,min_dist2,mean1,mean2 = min_dist_metric(data1,data2)
				metric.append(min_dist_between_subspaces + min_dist2)
			else:
				metric.append(np.inf)
		min_metric = min(metric)
		index_nearest_part = metric.index(min_metric)
		part_to_merge = p['neighbors'].iloc[index1][index_nearest_part]
			
		
		m_leaf,p = merge_two_polytopes(m_leaf,p,i,part_to_merge)
			
	return m_leaf,p 





def polytope_similarity(m_leaf,p,metric):
	
	part1 = []
	part2 = []
	score = []
	for i in range(len(p)):
		for j in p.iloc[i]['neighbors']:
			
			data1 = pd.DataFrame(m_leaf[i])#pd.DataFrame(m[p.iloc[i]['part_number']]) #m[p.iloc[i]['part_number']][2]
			data2 = pd.DataFrame(m_leaf[p[p['part_number']==j].index[0]])#pd.DataFrame(m[j]) #m[j][2]
			data1 = np.array(data1.drop('index',axis=1))
			data2 = np.array(data2.drop('index',axis=1))
				
			part1.append(p.iloc[i]['part_number'])
			part2.append(j)
			
			if metric == 'variance':
				var_ratio = variance_metric(data1,data2)
				score.append(var_ratio)
		
			if metric == 'centroid_ratio':
				ratio,difference  = centroid_metric(data1,data2)
				score.append(ratio)
 
			if metric == 'centroid_diff':
				ratio,difference  = centroid_metric(data1,data2)
				score.append(difference)
			
			if metric == 'min':
				min_dist_between_subspaces,mean,min_dist1,min_dist2,mean1,mean2 = min_dist_metric(data1,data2)
				diff = abs(min_dist_between_subspaces - mean)
				score.append(diff)
				
			if metric == 'min_corr':
				min_dist_between_subspaces,mean,min_dist1,min_dist2,mean1,mean2 = min_dist_metric(data1,data2)
				diff = abs(min_dist_between_subspaces - mean) + min_dist1 + min_dist2	
				score.append(diff)
				
	part_links = {'part1':part1,'part2':part2,'score':score}
	part_links = pd.DataFrame(part_links)
	part_links = part_links.sort_values(by='score',ascending=True)
	
	
	return part_links




def merging(m,part,metric):
	
	part_neigh = neighbors(part)
	
	p = part_neigh.copy(deep=True)
	p['part_number'] = p['part_number'].astype(float)
	p = p.query('leaf==True')
	p = p[['part_number','neighbors']]
	merged_part = []
	for i in range(len(p)):
		merged_part.append([])
	p['merged_part'] = merged_part
	p.index = np.arange(len(p))
	
	m_leaf = copy.deepcopy(m)
	m_leaf = np.delete(m_leaf, list(part_neigh.query('leaf==False')['part_number'])).tolist()

	
	m_leaf,p =  merge_single_data(m_leaf,p)
	#print(p)
	print('range of possible number of clusters: '+str(1)+'-'+str(len(p)))
	list_p = []
	list_m_leaf = []
	list_p.append(p)
	list_m_leaf.append(m_leaf)
	#G = nx.Graph()
	#for i in range(len(p)):
	#	G.add_node(p['part_number'].iloc[i])
	
	#c=0
	l = len(p)-1
	for i in range(l):
#	while nx.number_connected_components(G) > 1:
		#c+=1
		part_links = polytope_similarity(m_leaf,p,metric)
		#G.add_edge(part_links['part1'].iloc[0],part_links['part2'].iloc[0],weight=part_links['score'].iloc[0])
		m_leaf,p = merge_two_polytopes(m_leaf,p,part_links['part1'].iloc[0],part_links['part2'].iloc[0])
		
		list_p.append(p)
		list_m_leaf.append(m_leaf)
		#print(p)	
	 
	return list_m_leaf,list_p
