import numpy as np
import pandas as pd	
import polytope as pc

from Metrics import compute_metric



def data_splitting(data,cut_index,point_cut_distance):
			
	dist_positive = point_cut_distance[point_cut_distance['cut_index_'+str(int(cut_index))]>0]['point_index'].copy()
	dist_negative = point_cut_distance[point_cut_distance['cut_index_'+str(int(cut_index))]<=0]['point_index'].copy()
		
	data_pos = data.query('index=='+str(list(dist_positive))).copy()
	data_neg = data.query('index=='+str(list(dist_negative))).copy()
		
	return data_pos,data_neg




def cut_choice(data,cut_matrix,point_cut_distance,metric,exp):
	
	cut_matrix_reduced = cut_matrix.query('index1=='+str(list(data['index']))+' and index2=='+str(list(data['index'])))	.copy()
	cut_index = np.array(cut_matrix_reduced['cut_index'])
	
	metric_value_list = []
	for i in cut_index:
		data1,data2 = data_splitting(data,i,point_cut_distance)	
		metric_value = compute_metric(metric,data1,data2)
		metric_value_list.append(metric_value)
	 
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

 



def data_assignment(p1,p2,data,chosen_cut_index,point_cut_distance):
	
	data_pos,data_neg = data_splitting(data,chosen_cut_index,point_cut_distance)

	point_pos = data_pos.iloc[0].copy()
	point_pos = list(point_pos[0:-1])
	
	if (point_pos in p1) == True:
		data1 = data_pos.copy()
		data2 = data_neg.copy()
	else:
		data2 = data_pos.copy()
		data1 = data_neg.copy()
			
	return data1,data2





def recursive_process(p,data,cut_matrix,point_cut_distance,t0,lifetime,metric,exp):

	
	if len(data) <= 2: 
		return
	
	time_cut = np.random.exponential(1/p.volume)
	t0 += time_cut
	
	if t0 > lifetime:
		return

	try:
		hyperplane_direction,hyperplane_distance,chosen_cut_index  = cut_choice(data,cut_matrix,point_cut_distance,metric,exp)
	except TypeError:
		return
	
	p1,p2 = space_splitting(p,hyperplane_direction,hyperplane_distance)
		
	data1,data2 = data_assignment(p1,p2,data,chosen_cut_index,point_cut_distance) 

	part1 = [p1, data1]
	part2 = [p2, data2]
	part12 = [part1,part2]
	
	return part12,t0





def partitioning(cut_ensemble,t0,lifetime,metric,exp):

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
		#print('split ',count)

		try:

			father = i[1]
			part12,t0 = recursive_process(i[2],i[3],cut_matrix,point_cut_distance,i[0],lifetime,metric,exp)
			
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
				
			# se voglio fermarmi al primo taglio
			#if len(m)==3:
			#	break

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
	#m = [dict(i) for i in m]

	
	return m,part