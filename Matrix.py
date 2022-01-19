import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist



def cut_ensemble(X):

	
	# number of dimensions
	n_d = len(X[0]) 
	
	# index order 
	i1, i2 = np.tril_indices(len(X), k=-1) 
	
	# cut point  
	X_cut = []
	for i in range(n_d):
		x1 = np.take(X[:,i],i1)
		x2 = np.take(X[:,i],i2)
		X_cut.append((x1+x2)/2)
	cut_point =	np.array(X_cut).T
	
	# vector orthogonal to the cutting hyperplane 
	dist = cdist(X,X)
	cut_vector = []
	for i in range(n_d):
		d = np.subtract.outer(X[:,i],X[:,i]) 
		d_norm = d[i1, i2]/dist[i1, i2]
		cut_vector.append(d_norm)
	
	cut_vector = np.array(cut_vector).T
					
	magnitude_cut_vector = cut_vector*cut_point
	magnitude_cut_vector = np.sum(magnitude_cut_vector,axis=1)
	
	# point - hyperplane distance  (rows,columns = points,hyperplanes)
	point_cut_distance = cut_vector@X.T
	point_cut_distance = point_cut_distance.T - magnitude_cut_vector
	
	point_cut_distance = np.vstack([point_cut_distance.T,np.arange(len(point_cut_distance))]).T
	
	
	# output dataframe of indexed data
	index = np.arange(len(X))
	data = np.vstack([X.T,index]).T
	data_index = pd.DataFrame(data)
	columns = list(data_index.columns)
	columns[-1] = 'index'
	data_index.columns = [str(i) for i in columns]
	data_index['index'] = data_index['index'].astype(int)

	# output dataframe of hyperplanes 
	cut_matrix = np.vstack([i1, i2,np.arange(len(i1)),magnitude_cut_vector,cut_vector.T])
	cut_matrix = cut_matrix.T
	columns = ['index1','index2','cut_index','magnitude_norm_vect']
	for i in range(n_d):
		columns.append('norm_vect_'+str(i))
	cut_matrix = pd.DataFrame(cut_matrix)
	cut_matrix.columns = columns
	cut_matrix[['index1','index2','cut_index']] = cut_matrix[['index1','index2','cut_index']].astype(int)
	
	# output dataframe of cut - point distances
	columns = [*['cut_index_'+str(i) for i in range(len(cut_matrix))]]
	columns.append('point_index')
	point_cut_distance = pd.DataFrame(point_cut_distance)
	point_cut_distance.columns = columns
	point_cut_distance['point_index'] = point_cut_distance['point_index'].astype(int)
	
	
	return data_index,cut_matrix,point_cut_distance