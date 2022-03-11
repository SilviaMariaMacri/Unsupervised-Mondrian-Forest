import numpy as np
#import pandas as pd
from scipy.spatial.distance import cdist




def find_equal_columns(matrix):
	
	number_of_columns = len(matrix[0])
	column_index = np.array([],dtype=int)
	equal_column_index = np.array([],dtype=int)
	count = 0
	for i in range(number_of_columns):
		if i in column_index:
			continue
		comparison_i = (matrix[:,i][:,None] == matrix)
		count_elements = np.sum(comparison_i,axis=0)
		count_elements = np.where(count_elements!=0,count_elements,len(comparison_i))
		count_elements = np.where(count_elements==len(comparison_i),count_elements,0)
		equal_columns_i = np.argwhere(count_elements)
		equal_columns_i = equal_columns_i.reshape(1,len(equal_columns_i))[0]#.tolist()
		equal_column_index_i = (count*np.ones(len(equal_columns_i),dtype=int))#.tolist()
		
		column_index = np.concatenate((column_index,equal_columns_i))
		equal_column_index = np.concatenate((equal_column_index,equal_column_index_i))
		count += 1
		
	return column_index,equal_column_index






def cut_ensemble(X):
	
	'''
	Compute the cutting hyperplanes associated to the input dataset and, for 
	each of them, the sample-hyperplane distances
	
	Parameters:
	----------
	X : (n,m) array
		array of n points in m dimensions 

	Returns:
	-------
	data : array with n rows and m+1 columns
		it stores the indexed points (with indexes in the last column)
	cut_matrix : array 
		for each pair of points, it stores the information of the hyperplane 
		that separates them.
		The array columns store the following information:
		- column 1 and 2 : index of the two samples
		- column 3 : index of the hyperplane separating the two samples
		- column 4 : index of the hyperplane family (the hyperplanes dividing
	      the dataset in the same subgroups belong to the same family of 
		  hyperplanes, that is characterized by an index)
		- column 5 : magnitude of the normal vector that identifies the hyperplane
		- remaining columns : coordinates of the normal vector
	distance_matrix : array 
		it stores the sample-hyperplane distances with each row corresponding  
	    to a sample and each column to a hyperplane; the last column stores 
		the point indexes
	'''

	# drop double rows
	X_drop,idx = np.unique(X, axis=0, return_index=True)	
	X = X[np.sort(idx)].copy()
	
	# riduce dimensionalità nel caso in cui tutti i punti abbiano lo stesso  
	# valore nella stessa dimensione (altrimenti politopo ha volume nullo)
	dim_to_delete = []
	for i in range(len(X[0])):
		if len(np.unique(X[:,i])) == 1:
			dim_to_delete.append(i)
	X = np.delete(X,dim_to_delete,1)

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
	distance_matrix = np.vstack([point_cut_distance.T,np.arange(len(point_cut_distance))]).T
	
	# equivalent_hyperplanes
	splitted_data_matrix = point_cut_distance.copy()
	splitted_data_matrix = np.where(splitted_data_matrix<=0, False, splitted_data_matrix)
	splitted_data_matrix = np.where(splitted_data_matrix>0, True, splitted_data_matrix)
	splitted_data_matrix = splitted_data_matrix.astype(int)
	cut_index,equivalent_cut_index = find_equal_columns(splitted_data_matrix)
	eq_cut = np.hstack([cut_index.reshape(len(cut_index),1),equivalent_cut_index.reshape(len(cut_index),1)])
	eq_cut = eq_cut[eq_cut[:,0].argsort()]
	equivalent_cut = eq_cut[:,1].tolist()

	# hyperplane matrix
	cut_matrix = np.vstack([i1, i2,np.arange(len(i1)),equivalent_cut,magnitude_cut_vector,cut_vector.T])
	cut_matrix = cut_matrix.T 
	
	# indexed data
	data = np.hstack([X,np.arange(len(X)).reshape(len(X),1)])
	

	return data,cut_matrix,distance_matrix