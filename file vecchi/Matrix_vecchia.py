# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist



def DistanceMatrix(X):
	
	 
	
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
		print(i)
		
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








def DistanceMatrix_prova(X):
	
	
	
	if isinstance(X,pd.DataFrame):
		X = np.array(X)
	
	n_d = len(X[0]) # numero dimensioni
	
	
	dist = cdist(X,X)
	
	i1, i2 = np.tril_indices(len(X), k=-1) 
	
	# coppie di punti
	X1 = []
	X2 = []
	#X_cut = []
	for i in range(n_d):
		x1 = np.take(X[:,i],i1)
		x2 = np.take(X[:,i],i2)
		X1.append(x1)
		X2.append(x2)
		#X_cut.append((x1+x2)/2)

	coord1 = pd.DataFrame(X1).T
	coord1.columns = [*['x1_'+str(i) for i in range(n_d)]]

	coord2 = pd.DataFrame(X2).T
	coord2.columns = [*['x2_'+str(i) for i in range(n_d)]]
	
#	coord_cut = pd.DataFrame(X_cut).T
#	coord_cut.columns = [*['x_cut_'+str(i) for i in range(n_d)]]
	
	
	# versore del piano perpendicolare a distanza
	versori_coord = []
	for i in range(n_d):
		d = np.subtract.outer(X[:,i],X[:,i]) 
		d_norm = d[i1, i2]/dist[i1, i2]
		versori_coord.append(d_norm)
		
	versori_iperpiani_df = pd.DataFrame(versori_coord).T
	versori_iperpiani_df.columns = [*['norm_vect_cut_'+str(i) for i in range(n_d)]]
					
	#versori_iperpiani = np.array(versori_iperpiani_df)
	
#	magnitude_norm_vect = versori_iperpiani@(np.array(coord_cut).T)
#	magnitude_norm_vect = np.diag(magnitude_norm_vect)
	
	matrix = np.stack([i1.astype(int), i2.astype(int), dist[i1, i2]]).T#,magnitude_norm_vect
					
	matrix = pd.DataFrame(matrix)
	matrix.columns = ['index1','index2','dist']#,'magnitude_norm_vect']
	matrix['index1'] = matrix['index1'].astype(int)
	matrix['index2'] = matrix['index2'].astype(int)
	
	matrix['index_norm_vect'] = matrix.index

	for i in [coord1,coord2,#coord_cut,
		   versori_iperpiani_df]:
		matrix = pd.merge(matrix,i, left_index=True, right_index=True)

	
	'aggiunto'
	matrix['norm_vect_dist_0'] =  matrix['norm_vect_cut_1']
	matrix['norm_vect_dist_1'] =  -matrix['norm_vect_cut_0']
	
	versori_norm_dist = np.array(matrix[['norm_vect_dist_0','norm_vect_dist_1']])
	magnitude_norm_vect = versori_norm_dist@(np.array(coord1).T)
	magnitude_norm_vect = np.diag(magnitude_norm_vect)
	
	matrix['magnitude_norm_dist'] = magnitude_norm_vect

	#matrix['norm_vect_dist_1'] = 1 / np.sqrt(1 + matrix['norm_vect_cut_1']**2/matrix['norm_vect_cut_0']**2 )
	#matrix['norm_vect_dist_0'] = -matrix['norm_vect_cut_1']/matrix['norm_vect_cut_0'] *matrix['norm_vect_dist_1']


	matrix_copy = pd.DataFrame()
	for i in range(len(X)):
		print(i)
		
		for j in range(len(X[i])):
			matrix['point_'+str(j)] = X[i,j]
		matrix['point_index'] = i
		
		matrix_copy = pd.concat([matrix,matrix_copy])
	matrix = matrix_copy.copy()
	matrix.index = np.arange(len(matrix))

	'aggiunta (e manca altra parte)'
	i1_bis = np.array(matrix['index_norm_vect'])
	i2_bis = np.array(matrix['point_index'])
	
	# calcola distanza punti da congiungente la coppia di punti 
	dist_point_dist = versori_norm_dist@X.T
	dist_point_dist = dist_point_dist[i1_bis,i2_bis] - np.array(matrix['magnitude_norm_dist'])
	matrix['dist_point_dist'] = dist_point_dist
	
	#calcola modulo vettore normale a distanza della coppia per ogni altro punto
	versori_iperpiani = np.array(versori_iperpiani_df)
	magnitude_norm_point = versori_iperpiani@X.T
	magnitude_norm_point = magnitude_norm_point[i1_bis,i2_bis]
	matrix['magnitude_norm_point'] = magnitude_norm_point
	
	
	
	
	# intersezioni fra le rette
	
	dim0 = []
	dim1 = []
	
	for i in range(len(matrix)):
		print(i+1)
	
		a1 = matrix['norm_vect_cut_0'].iloc[i]
		b1 = matrix['norm_vect_cut_1'].iloc[i]
		c1 = matrix['magnitude_norm_point'].iloc[i]	
	
		a2 = matrix['norm_vect_dist_0'].iloc[i]
		b2 = matrix['norm_vect_dist_1'].iloc[i]
		c2 = matrix['magnitude_norm_dist'].iloc[i]	
		
		
		A = [[a1,b1],[a2,b2]]
		Ax = [[c1,b1],[c2,b2]]
		Ay = [[a1,c1],[a2,c2]]
			
		detA = np.linalg.det(A)
		detAx = np.linalg.det(Ax)
		detAy = np.linalg.det(Ay)
			
		# coordinate intersezione
		x = detAx/detA
		y = detAy/detA
			
		dim0.append(x)
		dim1.append(y)
	
	matrix['intersez_0'] = dim0
	matrix['intersez_1'] = dim1
	
	
	matrix = matrix[['index1', 'index2', 'x1_0', 'x1_1', 'x2_0', 'x2_1',
				   'index_norm_vect', 'dist', 'norm_vect_dist_0', 'norm_vect_dist_1',
				   'magnitude_norm_dist', 
				   'norm_vect_cut_0', 'norm_vect_cut_1', 
				   'point_index', 'point_0', 'point_1',
				   'dist_point_dist', 'magnitude_norm_point', 'intersez_0', 'intersez_1']]
	

	return matrix








# sbagliato - non usare
def DistanceMatrix_griglia_regolare(X):
	
	 
	
	if isinstance(X,pd.DataFrame):
		X = np.array(X)
	
	
	
	#coord_cut = np.array([[1,1],[1.5,1],[2,1],[1,0.8],[1,1],[1,1.2]])
	norm_vect_x = [1,1,1,0,0,0]
	norm_vect_y = [0,0,0,1,1,1]
	#versori_iperpiani =np.vstack([norm_vect_x,norm_vect_y]).T	
	#magnitude_norm_vect = versori_iperpiani@(np.array(coord_cut).T)
	#magnitude_norm_vect = np.diag(magnitude_norm_vect)
	magnitude_norm_vect = [1,1.5,2,0.8,1,1.2]

	matrix = {'norm_vect_0':norm_vect_x,'norm_vect_1':norm_vect_y,'magnitude_norm_vect':magnitude_norm_vect}
	matrix = pd.DataFrame(matrix)

	
	matrix['index_norm_vect'] = matrix.index


	matrix_copy = pd.DataFrame()
	for i in range(len(X)):
		print(i)
		
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


