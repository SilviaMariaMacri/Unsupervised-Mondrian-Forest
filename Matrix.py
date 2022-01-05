import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist



def distance_matrix(X):



	if isinstance(X,pd.DataFrame):
		X = np.array(X)
	
	n_d = len(X[0]) # numero dimensioni 
	
	
	dist = cdist(X,X)
	
	i1, i2 = np.tril_indices(len(X), k=-1) 
	
	# coppie di punti
	X1 = []
	X2 = []
	X_cut = []
	for i in range(n_d):
		x1 = np.take(X[:,i],i1)
		x2 = np.take(X[:,i],i2)
		X1.append(x1)
		X2.append(x2)
		X_cut.append((x1+x2)/2)

	coord1 = pd.DataFrame(X1).T
	coord1.columns = [*['x1_'+str(i) for i in range(n_d)]]

	coord2 = pd.DataFrame(X2).T
	coord2.columns = [*['x2_'+str(i) for i in range(n_d)]]
	
	coord_cut = pd.DataFrame(X_cut).T
	coord_cut.columns = [*['x_cut_'+str(i) for i in range(n_d)]]
	
	
	# versore piano perpendicolare a distanza
	versori_coord = []
	for i in range(n_d):
		d = np.subtract.outer(X[:,i],X[:,i]) 
		d_norm = d[i1, i2]/dist[i1, i2]
		versori_coord.append(d_norm)
		
	versori_iperpiani_df = pd.DataFrame(versori_coord).T
	versori_iperpiani_df.columns = [*['norm_vect_'+str(i) for i in range(n_d)]]
					
	versori_iperpiani = np.array(versori_iperpiani_df)
	
	
	magnitude_norm_vect = versori_iperpiani*np.array(coord_cut)
	magnitude_norm_vect = np.sum(magnitude_norm_vect,axis=1)
	#magnitude_norm_vect = np.diag(versori_iperpiani@(np.array(coord_cut).T))

	
	matrix = np.stack([i1.astype(int), i2.astype(int), dist[i1, i2],magnitude_norm_vect]).T
	matrix = pd.DataFrame(matrix)
	matrix.columns = ['index1','index2','dist','magnitude_norm_vect']
	matrix['index1'] = matrix['index1'].astype(int)
	matrix['index2'] = matrix['index2'].astype(int)
	
	matrix['index_norm_vect'] = matrix.index

	for i in [coord1,coord2,coord_cut,versori_iperpiani_df]:
		matrix = pd.merge(matrix,i, left_index=True, right_index=True)
	


	matrix_copy = pd.DataFrame()
	for i in range(len(X)):
		#print(i)
		
		for j in range(len(X[i])):
			matrix['point_'+str(j)] = X[i,j]
		matrix['point_index'] = i
		
		matrix_copy = pd.concat([matrix,matrix_copy])
	matrix = matrix_copy.copy()
	matrix.index = np.arange(len(matrix))

	i1_bis = np.array(matrix['index_norm_vect'])
	i2_bis = np.array(matrix['point_index'])
	
	
	dist_point_cut = versori_iperpiani@X.T
	dist_point_cut = dist_point_cut[i1_bis,i2_bis] - np.array(matrix['magnitude_norm_vect'])
	matrix['dist_point_cut'] = dist_point_cut
	
	
	
	return matrix
