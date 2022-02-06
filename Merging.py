import numpy as np
import pandas as pd
import copy
import polytope as pc 

from Metrics import compute_metric,min_dist_metric




def neighbors(part_leaf):
	
	'''
	Finds the adjacent polytopes of each leaf polytope obtained at the
	end of the partitioning procedure; two polytopes are considered 
	adjacent if they have at least a point in common
	
	Parameters:
	----------
	part_leaf : dataframe
		each row corresponds to a subspace, identified by a characteristic 
		number and a polytope.Polytope object
	
	Returns:
	-------
	p : dataframe
		each polytope is associated to a list of the characteristic numbers 
		of its neighbors; the lists of neighbors are stored in a column of
		the dataframe
	'''
	
	p = part_leaf.copy()
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



def merge_two_polytopes(p_init,m_leaf_init,part_to_remove,part_to_merge):
	
	'''
	Given as input a specific partition of the dataset/space, it merges two 
	polytopes, identified by their characteristic numbers 
	
	Parameters:
	----------
	p_init : dataframe
		it describes the current space partition; each row corresponds to a 
		polytope and the information about its characteristic number, its
		neighbors and the polytopes that have been already merged to them
		is stored   
	m_leaf_init : lists of dataframes
		each dataframe stores the points belonging to the polytope 
		identified by the same row index in p_init dataframe  
	part_to_remove : the characteristic number of the polytope that will be
		merged with one of its neighbors, changing its characteristic number
	part_to_merge : the characteristic number of the polytope that will be
		merged with one of its neighbors, preserving its characteristic number
	
	Returns:
	-------
	p : dataframe
		updated version of the input parameter p_init, after the merging
		of the two polytopes
	m_leaf : lists of dataframes
		updated version of the input parameter m_leaf_init, after the
		merging of the two datasets
	'''
	
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
	
	return p,m_leaf









#unisce partizione con solo un dato a quella più vicina
def merge_single_data(p_init,m_leaf_init): 
	
	'''
	Given as input a specific partition of the dataset/space, it merges each 
	polytope containing only one point with its nearest neighbor; 
	the degree of similarity/closeness is computed according to a metric  
	
	Parameters:
	----------
	p_init : dataframe
		it describes the current space partition; each row corresponds to a 
		polytope and the information about its characteristic number, its
		neighbors and the polytopes that have been already merged to them
		is stored   
	m_leaf_init : lists of dataframes
		each dataframe stores the points belonging to the polytope 
		identified by the same row index in p_init dataframe  
	
	Returns:
	-------
	p : dataframe
		updated version of the input parameter p_init, after the merging
		of the polytopes containing only one point
	m_leaf : lists of dataframes
		updated version of the input parameter m_leaf_init, after the
		merging of the single point datasets with the nearest ones
	'''
	
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
			
		
		p,m_leaf = merge_two_polytopes(p,m_leaf,i,part_to_merge)
			
	return p,m_leaf 





def polytope_similarity(p,m_leaf,metric):
	
	'''
	Computation of the similarity metric of each pair of neighboring polytopes
	
	Parameters:
	----------
	p : dataframe
		it describes the current space partition; each row corresponds to a 
		polytope   
	m_leaf : lists of dataframes
		each dataframe stores the points belonging to the polytope 
		identified by the same row index in p dataframe  
	metric : string identifying the chosen metric 
		('variance','centroid_ratio','centroid_diff','min','min_corr')
	
	Returns:
	-------	
	part_links : dataframe
		each row corresponds to a pair of neighboring polytopes, identified
		by their characteristic numbers; for each pair, the similarity
		metric value is stores; the metric values are shown in ascending order
	'''
	
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




def polytope_similarity_update(p,m_leaf,metric,removed_part,merged_part,part_links):
	
	'''
	Update of the dataframe storing the similarity metric values computed for
	each pair of polytopes, after the merging of two of them.
	
	Parameters:
	----------
	p : dataframe
		it describes the current space partition, after the merging of two
		polytopes  
	m_leaf : lists of dataframes
		each dataframe stores the points belonging to the polytope 
		identified by the same row index in p dataframe  
	metric : string identifying the chosen metric 
		('variance','centroid_ratio','centroid_diff','min','min_corr')
	removed_part : characteristic number of the polytope that has been merged
		to one of its neighbors, changing its characteristic number
	merged_part : characteristic number of the polytope that has been merged
		to one of its neighbors, preserving its characteristic number
	part_links : dataframe
		it stores the similarity metric values of each pair of polytopes 
		before the merging of two of them; each polytope is identified by
		its characteristic number
	
	Returns:
	-------	
	part_links_updated : dataframe
		update of the input part_links dataframe, after the merging of the
		two polytopes
	'''
	
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





def merging(part,m,metric):
	
	'''
	Given as input the final result of the partitioning phase, it progressively
	merges the more similar polytopes/subsets in order to obtain a hierarchical
	clusterization of the space. The similarity degree is determined by a metric
	
	Parameters:
	----------
	part : dataframe, output of Partitioning.partitioning  
		it stores the leaf polytopes obtained as final result of the partitioning  
	m : lists of dataframes, output of Partitioning.partitioning 
		each dataframes stores the point belonging to the polytope identified 
		by the same row index in part dataframe 
	metric : string identifying the chosen metric 
		('variance','centroid_ratio','centroid_diff','min','min_corr')
	
	Returns:
	-------	
	list_p : list of dataframes
		each dataframe describes a division of the space/dataset into a 
		specific number of clusters; the number of clusters in which the 
		dataset is divided is equal to the number of the polytopes 
		(number or dataframe rows); for each polytope, its characteristic 
		number, the list of neighbors and the list of merged polytopes are stored
	list_m_leaf : list of lists of dataframes
		each element of the list corresponds to an element  of list_p;
		it is a list of dataframes, each one storing the points belonging
		to the polytope identified by the same row index in p dataframe
	'''
	
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


	p,m_leaf =  merge_single_data(p,m_leaf)
	part_links = polytope_similarity(p,m_leaf,metric)
		
	print('range of possible number of clusters: '+str(1)+'-'+str(len(p)))
	list_p = []
	list_m_leaf = []
	list_p.append(p)
	list_m_leaf.append(m_leaf)
	
	l = len(p)-1
	for i in range(l):
		part1 = part_links['part1'].iloc[0]
		part2 = part_links['part2'].iloc[0]
		p,m_leaf = merge_two_polytopes(p,m_leaf,part1,part2)
		list_p.append(p)
		list_m_leaf.append(m_leaf)
		
		part_links = polytope_similarity_update(p,m_leaf,metric,part1,part2,part_links)
		 
	return list_p,list_m_leaf
