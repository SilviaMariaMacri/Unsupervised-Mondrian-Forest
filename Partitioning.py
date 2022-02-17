import numpy as np
import pandas as pd	
import polytope as pc

from Metrics import compute_metric



def data_splitting(data,cut_index,point_cut_distance):
	
	'''
	Divides the input dataset into two subsets separated by the input hyperplane
	
	Parameters:
	----------
	data : dataframe of indexed points
	cut_index : integer 
		it represents the index of the cutting hyperplane 
	point_cut_distance : output dataframe of Matrix.cut_ensemble
		it stores the sample-hyperplane distances
	Returns:
	-------
	data_pos,data_neg : dataframes of indexed points
		the points are divided on the basis of the sign of the point-hyperplane 
		distances (positive for data_pos and negative for data_neg) 
	'''
	
	dist_positive = point_cut_distance[point_cut_distance['cut_index_'+str(int(cut_index))]>0]['point_index'].copy()
	dist_negative = point_cut_distance[point_cut_distance['cut_index_'+str(int(cut_index))]<=0]['point_index'].copy()
		
	data_pos = data.query('index=='+str(list(dist_positive))).copy()
	data_neg = data.query('index=='+str(list(dist_negative))).copy()
	data_pos.index = np.arange(len(data_pos))
	data_neg.index = np.arange(len(data_neg))
		
	return data_pos,data_neg



def cut_choice(data,cut_matrix,point_cut_distance,metric,exp):
	
	'''
	Random extraction of the cutting hyperplane, with probability of extraction
	proportional to the similarity metric raised to a certain power
	
	Parameters:
	----------
	data : dataframe of indexed points
	cut_matrix : output dataframe of Matrix.cut_ensemble
		for each pair of points, it stores the information of the hyperplane 
		that separates them	  
	point_cut_distance : output dataframe of Matrix.cut_ensemble	 
		it stores the sample-hyperplane distances
	metric: string identifying the chosen metric 
		('variance','centroid_ratio','centroid_diff','min','min_corr')
	exp : power to which the metric is raised in order to obtain the 
		probability of extraction

	Returns:
	-------
	hyperplane_direction : hyperplane vector coordinates
	hyperplane_distance : hyperplane-origin distance
	chosen_cut_index : hyperplane index
	'''
	
	cut_matrix_reduced = cut_matrix.query('index1=='+str(list(data['index']))+' and index2=='+str(list(data['index'])))	.copy()
	equivalent_cut_index = np.array(cut_matrix_reduced['equivalent_cut_index'].unique())
	
	metric_value_list = np.array([],dtype=int)
	for i in equivalent_cut_index:
		#print(i)
		cut_matrix_reduced_i = cut_matrix_reduced.query('equivalent_cut_index == '+str(i)).copy()
		cut_index_i = cut_matrix_reduced_i.iloc[0]['cut_index']

		data1,data2 = data_splitting(data,cut_index_i,point_cut_distance)	
		metric_value = compute_metric(metric,data1,data2)
		metric_value_list = np.concatenate((metric_value_list,metric_value*np.ones((len(cut_matrix_reduced_i)))))
	
	cut_matrix_reduced = cut_matrix_reduced.sort_values(by='equivalent_cut_index')
	cut_index = np.array(cut_matrix_reduced['cut_index'])
	 
	if (len(set(metric_value_list)) == 1) and (np.isnan(metric_value_list[0])):
		return 
	metric_value_list = np.array(metric_value_list)
	index_to_remove = np.where(np.logical_or(np.isnan(metric_value_list),np.isinf(metric_value_list)))
	metric_value_list = np.delete(metric_value_list,index_to_remove)
	cut_index = np.delete(cut_index,index_to_remove)
		
	
	q = metric_value_list**exp
	weight = q/q.sum()
	try:
		chosen_cut_index = np.random.choice(cut_index,p=weight)
	except ValueError:
		q = metric_value_list
		weight = q/q.sum()
		chosen_cut_index = np.random.choice(cut_index,p=weight)

	chosen_cut = cut_matrix.query('cut_index=='+str(chosen_cut_index)).copy()
		
	n_d = len(data.iloc[0])-1
	names_norm_vect = []
	for i in range(n_d):
		names_norm_vect.append('norm_vect_'+str(i))
	hyperplane_direction = chosen_cut[names_norm_vect].iloc[0] 
	hyperplane_distance = chosen_cut['magnitude_norm_vect'].iloc[0]
	

	return hyperplane_direction,hyperplane_distance,chosen_cut_index




def space_splitting(p,hyperplane_direction,hyperplane_distance):
	
	'''
	Divides the input polytope into two subspaces separated by the input hyperplane
	
	Parameters:
	----------
	p : polytope.Polytope object
	hyperplane_direction : hyperplane vector coordinates (array or dataframe)
	hyperplane_distance : hyperplane-origin distance 
	
	Returns:
	-------
	p1,p2 : polytope.Polytope objects
	'''
	
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

 



def data_assignment(p1,p2,data_pos,data_neg):
	
	'''
	Given as input two polytopes and two datasets belonging to them, it assigns
	each dataset to the corresponding polytope.
	- All the points of each dataset must belong to the same polytope
	- The points lying on the common face of the two polytopes are not evaluated 
	  in order to assign the datasets to the polytopes
    - The case of all the points (of both datasets) belonging to the common
	  face is not considered, since the splitting procedure doesn't allow 
	  this situation
	
	Parameters:
	----------
	p1,p2 : polytope.Polytope objects
	data_pos,data_neg : dataframes of indexed points
		
	Returns:
	-------
	data1 : dataframe of indexed points belonging to p1
	data2 : dataframe of indexed points belonging to p2 
	'''

	for i in range(len(data_pos)):		
		point_pos = data_pos.iloc[i].copy()
		point_pos = list(point_pos[0:-1])
		
		if ((point_pos in p1) == True) and ((point_pos in p2) == False):
			data1 = data_pos.copy()
			data2 = data_neg.copy()
			break
		if ((point_pos in p2) == True) and ((point_pos in p1) == False):
			data2 = data_pos.copy()
			data1 = data_neg.copy()
			break
		if ((point_pos in p1) == True) and ((point_pos in p2) == True):
			for j in range(len(data_neg)):		
				point_neg = data_neg.iloc[j].copy()
				point_neg = list(point_neg[0:-1])
				
				if ((point_neg in p1) == True) and ((point_neg in p2) == False):
					data1 = data_neg.copy()
					data2 = data_pos.copy()
					break
				if ((point_neg in p2) == True) and ((point_neg in p1) == False):
					data2 = data_neg.copy()
					data1 = data_pos.copy()
					break
	
	return data1,data2


 


def recursive_process(p,data,cut_matrix,point_cut_distance,t0,lifetime,metric,exp):

	'''
	It takes as input a polytope and the subset of data belonging to it and 
	divides them into two parts giving as output the two new polytopes, the 
	corresponding subsets of data and the time when the cut is generated.
	It doesn't perform the split if the number of points contained in the 
	polytope is less or equal than two or if the time of the cut is higher 
	than the input lifetime parameter.

	Parameters:
	----------
	p : polytope.Polytope object
	data : dataframe of indexed points
 	cut_matrix : output dataframe of Matrix.cut_ensemble
		for each pair of points, it stores the information of the hyperplane 
		that separates them	  
	point_cut_distance : output dataframe of Matrix.cut_ensemble	 
		it stores the sample-hyperplane distances
	t0 : inital time
	lifetime : final time of the process
	metric: string identifying the chosen metric 
		('variance','centroid_ratio','centroid_diff','min','min_corr')
	exp : power to which the metric is raised in order to obtain the 
		probability of extraction
		
	Returns:
	-------
	p1,p2 : polytope.Polytope objects
	data1,data2 : dataframes of indexed points
		data1 belongs to p1 and data2 belongs to p2
	t0 : time of generation of the cut
	'''
	
	if len(data) <= 2: 
		return
	
	time_cut = np.random.exponential(1/len(data))#p.volume
	t0 += time_cut
	
	if t0 > lifetime:
		return

	try:
		hyperplane_direction,hyperplane_distance,chosen_cut_index  = cut_choice(data,cut_matrix,point_cut_distance,metric,exp)
	except TypeError:
		return
	
	p1,p2 = space_splitting(p,hyperplane_direction,hyperplane_distance)
	
	data_pos,data_neg = data_splitting(data,chosen_cut_index,point_cut_distance)	
	data1,data2 = data_assignment(p1,p2,data_pos,data_neg) 

	
	return p1,data1,p2,data2,t0





def partitioning(cut_ensemble,t0,lifetime,metric,exp):
	
	'''
	It performs the hierarchical splitting of the input dataset and its 
	underlying space. It takes as input the result of the Matrix.cut_ensemble
	function, that includes an adjustement of the considered dataset, and it 
	generates the underlying polytope as the smaller boundign box containing it.
	In order to generate the splits, it iterates the recursive_process function.
	It gives as output the complete history of the splitting.
	
	Parameters:
	----------
	cut_ensemble: lists of dataframes - complete output of Matrix.cut_ensemble
	t0 : inital time
	lifetime : final time of the process
	metric: string identifying the chosen metric 
		('variance','centroid_ratio','centroid_diff','min','min_corr')
	exp : power to which the metric is raised in order to obtain the 
		probability of extraction

	Returns:
	-------
	part : dataframe
		each row represents a polytope created during the hierarchical splitting;
		for each polytope, the dataframe stores the information about its 
		characteristic number, its creation time, the father polytope from 
		which it has been generated and if it is a leaf of the corresponding 
		tree structure.		
	m : list of dataframes
		each dataframe contains the points belonging to a specific polytope, 
		whose information is stored in each row of the part dataframe
	'''

	data = cut_ensemble[0]
	cut_matrix = cut_ensemble[1]
	point_cut_distance = cut_ensemble[2]
	
	n_d = len(data.columns)-1
	
	# initial space
	A_init_space = []
	b_init_space = []
	for i in range(n_d):
		A_i1 = list(np.zeros(n_d))
		A_i1[i] = 1.
		A_i2 = list(np.zeros(n_d))
		A_i2[i] = -1.
		A_init_space.append(A_i1)
		A_init_space.append(A_i2)

		length_i = data[str(i)].max() - data[str(i)].min()
		b_i1 = data[str(i)].max()+length_i*0.05
		b_i2 = -(data[str(i)].min()-length_i*0.05)
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
	
	time.append(t0)
	father_list.append('nan')
	part_number.append(count_part_number)
	polytope.append(p)
	

	
	count=0
	for i in m:
		count += 1
		

		try:

			father = i[1]
			p1,data1,p2,data2,t0 = recursive_process(i[2],i[3],cut_matrix,point_cut_distance,i[0],lifetime,metric,exp)
			
			count_part_number += 1
			m.append([t0,count_part_number,p1,data1])
			part_number.append(count_part_number)
			
			count_part_number += 1
			m.append([t0,count_part_number,p2,data2])
			part_number.append(count_part_number)
			
			time.extend((t0,t0))
			father_list.extend((father,father))
			polytope.extend((p1,p2))
				
			# se voglio fermarmi al primo taglio
			#if len(m)==3:
			#	break
		
			print('split ',count)

		except  TypeError:
			count -= 1
			continue
	
	print('total number of splits: '+str(count))
	
	part = {'time':time,'father':father_list,'part_number':part_number,'polytope':polytope}
	part = pd.DataFrame(part)

	leaf = []
	for i in range(len(part)):
		if part['part_number'].iloc[i] not in part['father'].unique():
			leaf.append(True)
		else:
			leaf.append(False)
	part['leaf'] = leaf

	part = part[['time', 'father', 'part_number', 'leaf', 'polytope']]
	
	m = list(np.array(m,dtype=object)[:,3])
	
	
	return part,m