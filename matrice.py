import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from itertools import combinations



def DistanceMatrix(X):
	
	 
	
	if isinstance(X,pd.DataFrame):
		X = np.array(X)
		
	
	#associo indice a ciascun punto
	n_d = len(X[0]) # numero dimensioni
	data = pd.DataFrame(X)
	data.columns = [*[f'dim{s}_point' for s in data.columns]]
	data['index_point'] = data.index


	#dataframe di coppie di punti
	pair_points = list(combinations(np.arange(len(X)), 2))
	pair_points = pd.DataFrame(pair_points)
	pair_points.columns = ['index_point_x','index_point_y']

	
	
	
	dist = pd.DataFrame(cdist(X,X))

	p1 = []
	p2 = []
	d = []
	for i,j in zip(pair_points['index_point_x'],pair_points['index_point_y']):
		p1.append(i)
		p2.append(j)
		d.append(dist.iloc[i,j])
	
		
	pair_points = {'index_point_x':p1,'index_point_y':p2,'dist':d}
	pair_points = pd.DataFrame(pair_points)

	
	#associo punti alle coppie di indici 
	pair_points = pd.merge(pair_points,data, how='left', left_on='index_point_x', right_on='index_point')
	pair_points = pair_points.drop('index_point',axis=1)
	pair_points = pd.merge(pair_points,data, how='left', left_on='index_point_y', right_on='index_point')
	pair_points = pair_points.drop('index_point',axis=1)
	
	pair_points = pair_points[['index_point_x', 'index_point_y', 'dim0_point_x', 'dim1_point_x', 'dim0_point_y', 'dim1_point_y', 'dist']]


	names = []
	#punto medio congiungente i due punti (cut)
	for i in range(n_d):
		pair_points['cut_point'+str(i)] = (pair_points['dim'+str(i)+'_point_x'] + pair_points['dim'+str(i)+'_point_y'])/2
		#vettore normale al taglio (parallelo a congiungente i due punti)
		pair_points['vettore_'+str(i)] = pair_points['dim'+str(i)+'_point_y'] - pair_points['dim'+str(i)+'_point_x']
		names.append('vettore_'+str(i))
	
	
	#normalizzo vettore normale
	for i in range(n_d):
		pair_points.loc[pair_points.index,'norm_vect_'+str(i)] = [*[(pair_points['vettore_'+str(i)].iloc[j]/np.linalg.norm(pair_points[names].iloc[j])) for j in range(len(pair_points)) ]]
	
	names_norm_vect = []
	names_cut = []
	for i in range(n_d):
		pair_points = pair_points.drop('vettore_'+str(i),axis=1)
		names_norm_vect.append('norm_vect_'+str(i))
		names_cut.append('cut_point'+str(i))
		
	#modulo vettore normale
	pair_points.loc[pair_points.index,'magnitude_norm_vect'] = [*[np.dot(pair_points[names_norm_vect].iloc[j],pair_points[names_cut].iloc[j]) for j in range(len(pair_points))]]
	


	dist_matrix = pd.DataFrame()
	
	for i in range(len(data)):
		
		matrix = pair_points.copy()
		
		for j in data:
			matrix[str(j)] = data[str(j)].iloc[i]
		#matrix = matrix.drop(matrix.query('index_point_x==index or index_point_y==index').index)
		#matrix = matrix.drop('index_point',axis=1)
		
		dist_matrix = pd.concat([dist_matrix,matrix])
		
	names_point = []	
	for i in range(n_d):
		names_point.append('dim'+str(i)+'_point')
		
	dist_matrix.index = np.arange(len(dist_matrix))
	
	
	#calcolo distanza punti da retta cut
	dist_matrix.loc[dist_matrix.index,'distance_point_cut'] = [*[(np.dot(dist_matrix[names_norm_vect].iloc[j],dist_matrix[names_point].iloc[j]) - dist_matrix['magnitude_norm_vect'].iloc[j]) for j in range(len(dist_matrix))]]
	






	return dist_matrix
