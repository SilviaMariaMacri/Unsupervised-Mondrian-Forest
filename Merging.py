import numpy as np
import pandas as pd
import copy
import polytope as pc 

from Metrics import compute_metric,min_dist_metric




def neighbors(p):
	
	neighbors_list = []
	for i in range(len(p)):
		poly_i = p['polytope'].iloc[i]
		neighbors = []
		for j in range(len(p)):
			if j == i:
				continue
			poly_j = p['polytope'].iloc[j]
			if (pc.is_adjacent(poly_i,poly_j) == True):
				neighbors.append(int(p['part_number'].iloc[j]))
		neighbors_list.append(neighbors)
		
	p['neighbors'] = neighbors_list			
	
	return p			



def merge_two_polytopes(m_leaf_init,p_init,part_to_remove,part_to_merge):
	
	p = p_init.copy(deep=True)
	m_leaf = copy.deepcopy(m_leaf_init)

	index1 = p[p['part_number']==part_to_remove].index[0]
	index2 = p[p['part_number']==part_to_merge].index[0]
	
	#unisco i dati in m_leaf
	data1 = m_leaf[index1].copy()
	data2 = m_leaf[index2].copy()
	data2 = pd.concat([data2,data1])
	data2.index = np.arange(len(data2))
	m_leaf[index2] = data2.copy()
	
	neigh = copy.deepcopy(list(p['neighbors']))
	merged_part = copy.deepcopy(list(p['merged_part']))

	#sostituisco vicini in tutte le partizioni di p e aggiungo vicini di single_part a part unita
	neigh[index1].remove(part_to_merge)
	neigh[index2].remove(part_to_remove)
	for j in neigh[index1]:
		if j not in neigh[index2]:
			neigh[index2].append(j)
		index_j = p[p['part_number']==j].index[0]
		neigh[index_j].remove(part_to_remove)
		if part_to_merge not in neigh[index_j]:
			neigh[index_j].append(part_to_merge)
			
	for j in merged_part[index1]:
		merged_part[index2].append(j)
	merged_part[index2].append(part_to_remove)
	
	p['neighbors'] = neigh
	p['merged_part'] = merged_part
	
	p = p.drop(index1)
	p.index = np.arange(len(p))
	m_leaf = np.delete(np.array(m_leaf,dtype=object), index1).tolist()
	
	return m_leaf,p









#unisce partizione con solo un dato a quella più vicina
def merge_single_data(m_leaf_init,p_init): 
	
	p = p_init.copy(deep=True)
	m_leaf = copy.deepcopy(m_leaf_init)

	list_part = list(p['part_number']).copy()
	for i in list_part:
		index1 = p[p['part_number']==i].index[0].copy()
		data1 = m_leaf[index1].copy()
		#considero solo partizioni con unico dato all'interno
		if len(data1) > 1:
			continue
		data1 = data1.drop('index',axis=1) 
		data1 = np.array(data1)
		#cerco minima distanza con part vicine
		metric = []
		for j in p['neighbors'].iloc[index1]:
			data2 = m_leaf[p[p['part_number']==j].index[0]].copy()
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
			
			data1 = m_leaf[i]
			data2 = m_leaf[p[p['part_number']==j].index[0]]
			metric_value = compute_metric(metric,data1,data2)
			score.append(metric_value)
			
			part1.append(p.iloc[i]['part_number'])
			part2.append(j)

	part_links = {'part1':part1,'part2':part2,'score':score}
	part_links = pd.DataFrame(part_links)
	part_links = part_links.sort_values(by='score',ascending=True)
	part_links.index = np.arange(len(part_links))
	part_links = part_links[part_links.index %2 == 0]
	part_links.index = np.arange(len(part_links))
	
	return part_links 




def polytope_similarity_update(m_leaf,p,metric,removed_part,merged_part,part_links):
	
	part_links = part_links.drop(part_links[part_links['part1']==removed_part].index)
	part_links = part_links.drop(part_links[part_links['part2']==removed_part].index)
	part_links = part_links.drop(part_links[part_links['part1']==merged_part].index)
	part_links = part_links.drop(part_links[part_links['part2']==merged_part].index)
	
	new_neighbors = list(p[p['part_number']==merged_part]['neighbors'])[0]
	merged_part_list = []
	score = []
	data1 = m_leaf[p[p['part_number']==merged_part].index[0]]
	for i in new_neighbors:
		data2 = m_leaf[p[p['part_number']==i].index[0]]
		metric_value = compute_metric(metric,data1,data2)
		score.append(metric_value)
		merged_part_list.append(merged_part)
	
	new_part_links = {'part1':merged_part_list,'part2':new_neighbors,'score':score}
	new_part_links = pd.DataFrame(new_part_links)
	
	part_links_updated = pd.merge(part_links,new_part_links, how='outer')
	part_links_updated = part_links_updated.sort_values(by='score',ascending=True)
	part_links_updated.index = np.arange(len(part_links_updated))
	
	return part_links_updated





def merging(m,part,metric):
	
	# riduco a partizione finale
	p = part.query('leaf==True').copy()
	p.index = np.arange(len(p))
	p = neighbors(p)
	merged_part = []
	for i in range(len(p)):
		merged_part.append([])
	p['merged_part'] = merged_part
	p = p[['part_number','neighbors','merged_part']].copy()
	
	m_leaf = copy.deepcopy(m)
	m_leaf = np.delete(np.array(m_leaf,dtype=object), list(part.query('leaf==False')['part_number'])).tolist()


	m_leaf,p =  merge_single_data(m_leaf,p)
	part_links = polytope_similarity(m_leaf,p,metric)
		
	print('range of possible number of clusters: '+str(1)+'-'+str(len(p)))
	list_p = []
	list_m_leaf = []
	list_p.append(p)
	list_m_leaf.append(m_leaf)
	
	l = len(p)-1
	for i in range(l):
		part1 = part_links['part1'].iloc[0]
		part2 = part_links['part2'].iloc[0]
		m_leaf,p = merge_two_polytopes(m_leaf,p,part1,part2)
		list_p.append(p)
		list_m_leaf.append(m_leaf)
		
		part_links = polytope_similarity_update(m_leaf,p,metric,part1,part2,part_links)
		 
	return list_m_leaf,list_p
