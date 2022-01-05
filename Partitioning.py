import numpy as np
import polytope as pc
import pypoman
from numpy.random import choice
import pandas as pd	
from itertools import combinations
import math

from Metrics import variance_metric,centroid_metric,min_dist_metric



#metric = variance, centroid_diff,centroid_ratio, min, min_corr


def cut_choice(dist_matrix,data,metric,exp):

	data_pair = list(combinations(data['index'], 2))
	data_pair = pd.DataFrame(data_pair)
	data_pair.columns = ['index2','index1']


	n_d = len(data.columns)-1   
	names = []
	for i in range(n_d):
		names.append('point_'+str(i))
	

	metric_value = []
	for i in range(len(data_pair)):
		#print(i)
		
		matrix = dist_matrix[(dist_matrix['index1']==data_pair['index1'].iloc[i]) & (dist_matrix['index2']==data_pair['index2'].iloc[i])].copy()

		data1 = np.array(matrix.query('dist_point_cut>0')[names].copy())
		data2 = np.array(matrix.query('dist_point_cut<=0')[names].copy())
		
		#if (len(data1)==1) and (len(data2)==1):
		#	metric_value.append('nan')
			
		#else:
			
			
			
		if metric == 'variance':	
			var_ratio = variance_metric(data1,data2)
			metric_value.append(var_ratio)
			
		if metric == 'centroid_diff':	
			ratio,difference = centroid_metric(data1, data2)
			metric_value.append(difference)

		if metric == 'centroid_ratio':	
			ratio,difference = centroid_metric(data1, data2)
			metric_value.append(ratio)
			
		if metric == 'min':
			min_dist_between_subspaces,media,min_dist1,min_dist2,mean1,mean2 = min_dist_metric(data1,data2)
			diff = abs(min_dist_between_subspaces - media)
			metric_value.append(diff)
			
		if metric == 'min_corr':	
			min_dist_between_subspaces,media,min_dist1,min_dist2,mean1,mean2 = min_dist_metric(data1,data2)
			if len(data1) == 1:
				diff = abs(min_dist_between_subspaces - media) + min_dist2
			if len(data2) == 1:
				diff = abs(min_dist_between_subspaces - media) + min_dist1 
			else:
				diff = abs(min_dist_between_subspaces - media) + min_dist1 + min_dist2
			metric_value.append(diff)
		
	data_pair['metric'] = metric_value


	if (len(data_pair['metric'].unique()) == 1) and (data_pair['metric'].unique()[0]=='nan'):
		return 

	data_pair = data_pair.drop(data_pair[data_pair['metric']=='nan'].index)
	data_pair = data_pair.drop(data_pair[data_pair['metric']==np.inf].index)
	
	data_pair.index = np.arange(len(data_pair))

	q=data_pair['metric']**exp
	weight = q/q.sum()
	try:
		index_cut = choice(data_pair.index,p=weight)
	except ValueError:
		q=data_pair['metric']
		weight = q/q.sum()
		index_cut = choice(data_pair.index,p=weight)

	matrix = dist_matrix[(dist_matrix['index1']==data_pair['index1'].iloc[index_cut]) & (dist_matrix['index2']==data_pair['index2'].iloc[index_cut])].copy()
	matrix.index = np.arange(len(matrix))
	
	names_norm_vect = []
	for i in range(n_d):
		names_norm_vect.append('norm_vect_'+str(i))
	hyperplane_direction = matrix[names_norm_vect].iloc[0] 
	hyperplane_distance = matrix['magnitude_norm_vect'].iloc[0]
	
	names.append('point_index')
	names.append('dist_point_cut')

	matrix = matrix[names]



	return hyperplane_direction,hyperplane_distance,matrix 




def space_splitting(p,hyperplane_direction,hyperplane_distance):
	
	
	A_hyperplane_1 = list(hyperplane_direction) 
	b_hyperplane_1 = hyperplane_distance  
	A_hyperplane_2 = list(-hyperplane_direction) 
	b_hyperplane_2 = -hyperplane_distance


		
	A1 = list(p.A)
	A1.append(A_hyperplane_1)
	b1 = list(p.b)
	b1.append(b_hyperplane_1)
	p1 = pc.Polytope(np.array(A1), np.array(b1))
	p1 = pc.reduce(p1)
	
	A2 = list(p.A)
	A2.append(A_hyperplane_2)
	b2 = list(p.b)
	b2.append(b_hyperplane_2)
	p2 = pc.Polytope(np.array(A2), np.array(b2))
	p2 = pc.reduce(p2)
	


	return p1,p2







def data_assignment(p1,p2,matrix):
	
	columns = list(np.arange(len(matrix.columns)-2))
	columns.append('index')
	columns.append('dist_point_cut')
	matrix.columns = columns#['index' if x=='point_index' else x for x in matrix.columns]
	
	point = list(matrix.query('dist_point_cut>0').iloc[0].copy()[:-2])
	
	if (point in p1) == True:
		data1 = matrix.query('dist_point_cut>0').copy()
		data2 = matrix.query('dist_point_cut<=0').copy()
	else:
		data2 = matrix.query('dist_point_cut>0').copy()
		data1 = matrix.query('dist_point_cut<=0').copy()
			

	
	data1.index = np.arange(len(data1))
	data1 = data1.drop('dist_point_cut',axis=1)
	data2.index = np.arange(len(data2))
	data2 = data2.drop('dist_point_cut',axis=1)
	
	
	return data1,data2






def recursive_process(t0,lifetime,dist_matrix,p,data,metric,exp):

	
	if len(data) <= 2: 
		return
	
	
	
	# genera tempo per cut 
	time_cut = np.random.exponential(1/p.volume)


	t0 += time_cut
	
	if t0 > lifetime:
		return



	#riduce matrice delle distanze ai punti contenuti nel politopo
	dist_matrix = pd.merge(dist_matrix,data[data['index']!=dist_matrix['index1'].unique()[len(dist_matrix['index1'].unique())-1]], how='right', left_on='index2', right_on='index')
	dist_matrix = dist_matrix.drop(data.columns,axis=1)
	dist_matrix = pd.merge(dist_matrix,data[data['index']!=0], how='right', left_on='index1', right_on='index')
	dist_matrix = dist_matrix.drop(data.columns,axis=1)
	dist_matrix = pd.merge(dist_matrix,data, how='right', left_on='point_index', right_on='index')
	dist_matrix = dist_matrix.drop(data.columns,axis=1)
	

	try:
		hyperplane_direction,hyperplane_distance,matrix  = cut_choice(dist_matrix,data,metric,exp)
	except TypeError:
		return
	
	p1,p2 = space_splitting(p,hyperplane_direction,hyperplane_distance)
		
	data1,data2 = data_assignment(p1,p2,matrix) 


	
	part1 = [p1, data1]
	part2 = [p2, data2]
	part12 = [part1,part2]
	
	return part12,t0





def partitioning(X,t0,lifetime,dist_matrix,metric,exp):


	data = pd.DataFrame(X)
	data['index'] = data.index
	
	n_d = len(X[0])
	
	A_init_space = []
	b_init_space = []
	for i in range(n_d):
		A_i1 = list(np.zeros(n_d))
		A_i1[i] = 1.
		A_i2 = list(np.zeros(n_d))
		A_i2[i] = -1.
		A_init_space.append(A_i1)
		A_init_space.append(A_i2)

		length_i = data[i].max() - data[i].min()
		b_i1 = data[i].max()+length_i*0.05
		b_i2 = -(data[i].min()-length_i*0.05)
		b_init_space.append(b_i1)		
		b_init_space.append(b_i2)
		
	p = pc.Polytope(np.array(A_init_space), np.array(b_init_space))



	m=[]
	count_part_number = 0 	
	m0 = [ t0,count_part_number,p,data ] 
	m.append(m0)
	
	time = []
	father_list = []
	part_number = []
	polytope = []
	vertices = []
	
	time.append(t0)
	father_list.append('nan')
	part_number.append(count_part_number)
	polytope.append(p)
	#if (n_d==2) or (n_d==3):
	vert = pypoman.compute_polytope_vertices(np.array(A_init_space), np.array(b_init_space))
	vertices.append(vert)
	

	
	count=0
	for i in m:
		
		
		count += 1
		#print('split ',count)

		try:

			father = i[1]
			part12,t0 = recursive_process(i[0],lifetime,dist_matrix,i[2],i[3],metric,exp)
			
			count_part_number += 1
			m.append([t0,count_part_number,part12[0][0],part12[0][1]])
			part_number.append(count_part_number)
			
			count_part_number += 1
			m.append([t0,count_part_number,part12[1][0],part12[1][1]])
			part_number.append(count_part_number)
			

			
			for j in range(2):
				time.append(t0)
				father_list.append(father)
				poly = part12[j][0]
				polytope.append(poly)
				vert = pypoman.compute_polytope_vertices(poly.A,poly.b)
				# ordino vertici: (per più dimensioni da errore?)
				# compute centroid
				cent=(sum([v[0] for v in vert])/len(vert),sum([v[1] for v in vert])/len(vert))
				# sort by polar angle
				vert.sort(key=lambda v: math.atan2(v[1]-cent[1],v[0]-cent[0]))
				vertices.append(vert)
				
			# se voglio fermarmi al primo taglio
			#if len(m)==3:
			#	break
			
		except  TypeError:
			count -= 1
			continue
	
	print('total number of splits: '+str(count))
	
	part = {'time':time,'father':father_list,'part_number':part_number,'polytope':polytope,'box':vertices}
		
	part = pd.DataFrame(part)


	leaf = []
	for i in range(len(part)):
		if part['part_number'].iloc[i] not in part['father'].unique():
			leaf.append(True)
		else:
			leaf.append(False)


		
	part['leaf'] = leaf

	part = part[['time', 'father', 'part_number', 'leaf', 'polytope','box'	]]
	
	
	
	
	return m,part
