import numpy as np
import pandas as pd
import copy
import polytope as pc 
from scipy.spatial.distance import cdist

from Metrics import compute_metric


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
				neighbors.append(int(p['id_number'].iloc[j]))
		neighbors_list.append(neighbors)
		
	return neighbors_list		

		

def merge_two_polytopes(poly_number_input,neigh_poly_input,merged_poly_input,merg_data_input,part_to_remove,part_to_merge):
	
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
	
	poly_number = copy.deepcopy(poly_number_input)
	neigh_poly = copy.deepcopy(neigh_poly_input)
	merged_poly = copy.deepcopy(merged_poly_input)
	merg_data  = copy.deepcopy(merg_data_input)

	index1 = poly_number.index(part_to_remove)
	index2 = poly_number.index(part_to_merge)
	
	#unisco i dati in merg_data
	data1 = merg_data[index1].copy()
	data2 = merg_data[index2].copy()
	data2 = np.hstack([data1,data2])
	merg_data[index2] = data2.copy()
	merg_data.pop(index1)
	
	neigh_poly[index1].remove(part_to_merge)
	neigh_poly[index2].remove(part_to_remove)
	for j in neigh_poly[index1]:
		if j not in neigh_poly[index2]:
			neigh_poly[index2].append(j)
		index_j = poly_number.index(j)
		neigh_poly[index_j].remove(part_to_remove)
		if part_to_merge not in neigh_poly[index_j]:
			neigh_poly[index_j].append(part_to_merge)
	neigh_poly.pop(index1)
	
	for j in merged_poly[index1]:
		merged_poly[index2].append(j)
	merged_poly[index2].append(part_to_remove)
	merged_poly.pop(index1)
	
	poly_number.pop(index1)
	
	return poly_number,neigh_poly,merged_poly,merg_data 






#unisce partizione con solo un dato a quella più vicina
def merge_single_data(poly_number_input,neigh_poly_input,merged_poly_input,merg_data_input,data): 
	
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
	
	poly_number = copy.deepcopy(poly_number_input)
	neigh_poly = copy.deepcopy(neigh_poly_input)
	merged_poly = copy.deepcopy(merged_poly_input)
	merg_data = copy.deepcopy(merg_data_input)
	
	poly_number_fixed = copy.deepcopy(poly_number)
	for i in poly_number_fixed:
		poly_index_i = poly_number.index(i)
		data1_index = merg_data[poly_index_i].copy()
		if len(data1_index) > 1:
			continue
		data1 = data[data1_index].copy()
		# metrica = distanza minima fra gruppi
		metric = []
		for j in neigh_poly[poly_index_i]:
			poly_index_j = poly_number.index(j)
			data2_index = merg_data[poly_index_j].copy()
			data2 = data[data2_index].copy()
			min_dist = min(cdist(data1,data2)[0])
			metric.append(min_dist)
		min_metric = min(metric)
		index_nearest_part = metric.index(min_metric)
		
		part_to_remove = i
		part_to_merge = neigh_poly[poly_index_i][index_nearest_part]
		poly_number,neigh_poly,merged_poly,merg_data = merge_two_polytopes(poly_number,neigh_poly,merged_poly,merg_data,part_to_remove,part_to_merge)
			
	return poly_number,neigh_poly,merged_poly,merg_data 





def polytope_similarity(poly_number,neigh_poly,merg_data,metric,data):
	
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
	for i in range(len(poly_number)):
		for j in neigh_poly[i]:
			
			data1 = data[merg_data[i]].copy()
			data2 = data[merg_data[poly_number.index(j)]].copy()
			metric_value = compute_metric(metric,data1,data2)
			score.append(metric_value)
			
			part1.append(poly_number[i])
			part2.append(j)

	part_links = {'part1':part1,'part2':part2,'score':score}
	part_links = pd.DataFrame(part_links)
	part_links = part_links.sort_values(by='score',ascending=True)
	part_links.index = np.arange(len(part_links))
	part_links = part_links[part_links.index %2 == 0]
	part_links.index = np.arange(len(part_links))
	
	return part_links 




def polytope_similarity_update(poly_number,neigh_poly,merg_data,metric,removed_part,merged_part,part_links,data):
	
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
	
	new_neighbors = neigh_poly[poly_number.index(merged_part)]
	merged_part_list = []
	score = []
	data1 = data[merg_data[poly_number.index(merged_part)]].copy()
	if isinstance(new_neighbors,list):
		for i in new_neighbors:
			data2 = data[merg_data[poly_number.index(i)]].copy()
			metric_value = compute_metric(metric,data1,data2)
			score.append(metric_value)
			merged_part_list.append(merged_part)		
	else:	
		data2 = data[merg_data[poly_number.index(new_neighbors)]].copy()
		metric_value = compute_metric(metric,data1,data2)
		score.append(metric_value)
		merged_part_list.append(merged_part)
	
	new_part_links = {'part1':merged_part_list,'part2':new_neighbors,'score':score}
	new_part_links = pd.DataFrame(new_part_links)
	
	part_links_updated = pd.merge(part_links,new_part_links, how='outer')
	part_links_updated = part_links_updated.sort_values(by='score',ascending=True)
	part_links_updated.index = np.arange(len(part_links_updated))
	
	return part_links_updated





def class_assignment(m_leaf):

	classes = np.array([])
	indices = np.array([])
	for j in range(len(m_leaf)):
		classes_j = j*np.ones(len(m_leaf[j]),dtype=int)
		classes = np.hstack([classes,classes_j])
		indices_j = np.array(m_leaf[j]).copy()
		indices = np.hstack([indices,indices_j])
	df = {'index':indices,'class':classes}
	df = pd.DataFrame(df)
	df = df.sort_values(by='index').astype(int)
	classified_data = list(df['class'])
		
	return classified_data






def merging(part_space,part_data,metric,data):
	
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
	
	data = data[:,:-1]
	
	# riduco part_space e part_data a partizione finale
	part_space_leaf = part_space.query('leaf==True').copy()
	part_data_leaf = copy.deepcopy(part_data)
	part_data_leaf = np.delete(np.array(part_data_leaf,dtype=object), 
							   part_space.query('leaf==False').index)
	
	poly_number = part_space_leaf['id_number'].tolist()
	neigh_poly = neighbors(part_space_leaf)
	merged_poly = [[] for _ in range(len(part_space_leaf))]
	
	poly_number,neigh_poly,merged_poly,merg_data = merge_single_data(poly_number,neigh_poly,merged_poly,part_data_leaf.tolist(),data)
	
	merg_space = {'id_number':poly_number,'neighbors':neigh_poly,'merged':merged_poly}
	merg_space = pd.DataFrame(merg_space)	
		
	print('range of possible number of clusters: '+str(1)+'-'+str(len(merg_space)))
	merg_space_list = []
	merg_data_list = []
	merg_space_list.append(merg_space)
	classified_data = class_assignment(merg_data)
	merg_data_list.append(classified_data)
	
	
	part_links = polytope_similarity(poly_number,neigh_poly,merg_data,metric,data)
	
	l = len(poly_number)-1
	for i in range(l):
		
		part1 = part_links['part1'].iloc[0]
		part2 = part_links['part2'].iloc[0]
		poly_number,neigh_poly,merged_poly,merg_data = merge_two_polytopes(poly_number,neigh_poly,merged_poly,merg_data,part1,part2)
		part_links = polytope_similarity_update(poly_number,neigh_poly,merg_data,metric,part1,part2,part_links,data)
		
		merg_space = {'id_number':poly_number,'neighbors':neigh_poly,'merged':merged_poly}
		merg_space = pd.DataFrame(merg_space)
		
		merg_space_list.append(merg_space)
		classified_data_i = class_assignment(merg_data)
		merg_data_list.append(classified_data_i)
		
	return merg_space_list,merg_data_list
