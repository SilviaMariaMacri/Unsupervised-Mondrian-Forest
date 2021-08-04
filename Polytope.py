import numpy as np
import polytope as pc


A = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [-1.0, -0.0],
              [-0.0, -1.0]])

b = np.array([2.0, 1.0, 0.0, 0.0])

p1 = pc.Polytope(A, b)


# oppure
#p1 = pc.box2poly([[0, 2], [0, 1]])

'''
print(p1)
Single polytope 
  [[ 1.  0.] |    [[2.]
   [ 0.  1.] x <=  [1.]
   [-1. -0.] |     [0.]
   [-0. -1.]]|     [0.]]
  '''
  
#righe = iperpiani
#colonne = numero di dimensioni


#vedere se punto è contenuto all'interno del politopo
[0.5, 0.5] in p1

p2 = pc.box2poly([[0, 1], [0, 0.5]])

p1 <= p2  #p1 subset of p2
p1 == p2  #idem come sopra e viceversa
p1.union(p2)
p1.diff(p2)
p1.intersect(p2)



#%%
fig,ax = plt.subplots()
#p1.plot()
#p1.bounding_box()
p2.plot()
p2.bounding_box()

p.plot()



#%%

import numpy as np
import polytope as pc
import pypoman
from numpy.random import choice
import pandas as pd	
from itertools import combinations
from scipy.spatial.distance import pdist










def CutMinDist(dist_matrix,data):
	
	data_pair = list(combinations(data['index'], 2))
	data_pair = pd.DataFrame(data_pair)
	data_pair.columns = ['index2','index1']


	n_d = len(data.columns)-1   #forse puoi fare una classe
	names = []
	for i in range(n_d):
		names.append('point_'+str(i))
	
	diff_min_dist = []
	for i in range(len(data_pair)):
		#print(i)
		
		matrix = dist_matrix[(dist_matrix['index1']==data_pair['index1'].iloc[i]) & (dist_matrix['index2']==data_pair['index2'].iloc[i])].copy()

		data1 = np.array(matrix.query('dist_point_cut>0')[names].copy())
		data2 = np.array(matrix.query('dist_point_cut<0')[names].copy())
		
		if (len(data1)==1) or (len(data2)==1):
			diff_min_dist.append('nan')
			
		else:
			
			pd1 = cdist(data1,data1)
			pd2 = cdist(data2,data2)
			
			min1 = np.min(np.where(pd1!= 0, pd1, np.inf),axis=0)
			min2 = np.min(np.where(pd2!= 0, pd2, np.inf),axis=0)
			
			min_tot = np.hstack([min1,min2])
			media = np.mean(min_tot)
		
			dist = cdist(data1,data2)
			min_dist_fra_partizioni = dist.min()

			#media,min_dist_fra_partizioni = MinDist(data1,data2,matrix)
			diff = min_dist_fra_partizioni - media
			diff_min_dist.append(diff)
		
	data_pair['diff_min_dist'] = diff_min_dist
	

	if data_pair['diff_min_dist'].unique()[0]=='nan':
		return 

	data_pair = data_pair.drop(data_pair[data_pair['diff_min_dist']=='nan'].index)
	
	data_pair.index = np.arange(len(data_pair))

	q=data_pair['diff_min_dist']**50
	weight = q/q.sum()
	index_cut = choice(data_pair.index,p=weight)
	
	
	matrix = dist_matrix[(dist_matrix['index1']==data_pair['index1'].iloc[index_cut]) & (dist_matrix['index2']==data_pair['index2'].iloc[index_cut])].copy()
	matrix.index = np.arange(len(matrix))
	
	names_norm_vect = []
	for i in range(n_d):
		names_norm_vect.append('norm_vect_'+str(i))
	A_hyperplane_1 = list(matrix[names_norm_vect].iloc[0]) 
	b_hyperplane_1 = matrix['magnitude_norm_vect'].iloc[0]  
	A_hyperplane_2 = list(-matrix[names_norm_vect].iloc[0]) 
	b_hyperplane_2 = -matrix['magnitude_norm_vect'].iloc[0]


	names.append('point_index')
	names.append('dist_point_cut')

	matrix = matrix[names]



	return A_hyperplane_1,b_hyperplane_1,A_hyperplane_2,b_hyperplane_2,matrix 





def FindDataPartition(p1,p2,matrix):
	
	columns = list(np.arange(len(matrix.columns)-2))
	columns.append('index')
	columns.append('dist_point_cut')
	matrix.columns = columns#['index' if x=='point_index' else x for x in matrix.columns]
	
	point = list(matrix.query('dist_point_cut>0').iloc[0].copy()[:-2])
	
	if (point in p1) == True:
		data1 = matrix.query('dist_point_cut>0').copy()
		data2 = matrix.query('dist_point_cut<0').copy()
	else:
		data2 = matrix.query('dist_point_cut>0').copy()
		data1 = matrix.query('dist_point_cut<0').copy()
			
	
	
	data1.index = np.arange(len(data1))
	data1 = data1.drop('dist_point_cut',axis=1)
	data2.index = np.arange(len(data2))
	data2 = data2.drop('dist_point_cut',axis=1)
	
	
	return data1,data2







def Cut_with_data(data,p,dist_matrix):
	
	
	
	try:
		A_hyperplane_1,b_hyperplane_1,A_hyperplane_2,b_hyperplane_2,matrix  = CutMinDist(dist_matrix,data)
	except TypeError:
		return
	
	
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
	
	
	data1,data2 = FindDataPartition(p1,p2,matrix)

	
	
	return p1,p2,data1,data2












#Mondrian_SingleCut(i[0],lifetime,i[1],i[2],dist_matrix,i[3],i[4])
def Mondrian_SingleCut(t0,lifetime,p,data,dist_matrix,father,neighbors):

	

	# genera tempo per cut 
	time_cut = np.random.exponential(1/p.volume)


	t0 += time_cut
	
	if t0 > lifetime:
		return
	
	
	if len(data) <= 2: 
		return
	
	
	
	#riduce matrice delle distanze ai punti contenuti nel politopo
	dist_matrix = pd.merge(dist_matrix,data[data['index']!=dist_matrix['index1'].unique()[len(dist_matrix['index1'].unique())-1]], how='right', left_on='index2', right_on='index')
	dist_matrix = dist_matrix.drop(data.columns,axis=1)
	dist_matrix = pd.merge(dist_matrix,data[data['index']!=0], how='right', left_on='index1', right_on='index')
	dist_matrix = dist_matrix.drop(data.columns,axis=1)
	dist_matrix = pd.merge(dist_matrix,data, how='right', left_on='point_index', right_on='index')
	dist_matrix = dist_matrix.drop(data.columns,axis=1)
	
	try:
		p1,p2,data1,data2 = Cut_with_data(data,p,dist_matrix)
	except TypeError:
		return

	neigh1 = [p2]
	neigh2 = [p1]
	for i in neighbors:
		if pc.is_adjacent(p1,i) ==True:
			neigh1.append(i)
		if pc.is_adjacent(p2,i) ==True:
			neigh2.append(i)
		
	
	
	part1 = [t0, p1, data1, neigh1]
	part2 = [t0, p2, data2, neigh2]
	risultato = [part1, part2, t0, father]
	
	
	
	
	return risultato











def Mondrian(X,t0,lifetime,dist_matrix):


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



	neighbors = []



	m=[]
	count_part_number = 0
	m0 = [ t0,p,data,count_part_number,neighbors ] 
	m.append(m0)
	
	time = []
	father = []
	part_number = []
	polytope = []
	vertices = []
	
	time.append(t0)
	father.append('nan')
	part_number.append(count_part_number)
	polytope.append(p)
	vert = pypoman.compute_polytope_vertices(np.array(A_init_space), np.array(b_init_space))
	vertices.append(vert)


	
	count=0
	for i in m:
		
		
		count += 1
		print('iterazione: ',count)

	
		try:

			mondrian = Mondrian_SingleCut(i[0],lifetime,i[1],i[2],dist_matrix,i[3],i[4])
			
			count1 = count_part_number+1
			m.append([mondrian[0][0],mondrian[0][1],mondrian[0][2],count_part_number+1,mondrian[0][3]])
			count_part_number += 1
			part_number.append(count_part_number)
			
			count2 = count_part_number+1
			m.append([mondrian[1][0],mondrian[1][1],mondrian[1][2],count_part_number+1,mondrian[1][3]])
			count_part_number += 1
			part_number.append(count_part_number)
			
			
			for j in range(2):
				time.append(mondrian[2])
				father.append(mondrian[3])
				poly = mondrian[j][1]
				polytope.append(poly)
				vert = pypoman.compute_polytope_vertices(poly.A,poly.b)
				vertices.append(vert)
				
			# se voglio fermarmi al primo taglio
			#if len(m)==3:
			#	break
			

		except  TypeError:
			continue
		
	
	neigh_part = []	
	for i in m:
		neigh_part.append(i[4])

	part = {'time':time,'father':father,'part_number':part_number,'polytope':polytope,'neighbors':neigh_part,'box':vertices}
	part = pd.DataFrame(part)
	

	leaf = []
	for i in range(len(part)):
		if part['part_number'].iloc[i] not in part['father'].unique():
			leaf.append(True)
		else:
			leaf.append(False)


		
	part['leaf'] = leaf

	part = part[['time', 'father', 'part_number', 'polytope', 'neighbors', 'leaf','box']]	
	
	
	
	
	return m,part



#%%

from scipy.spatial import ConvexHull, convex_hull_plot_2d

hull = ConvexHull(part_leaf['box'].iloc[2])

convex_hull_plot_2d(hull)

part_leaf = part.query('leaf==True')
part_leaf.index = np.arange(len(part_leaf))
#%%	  non serve credo ma ricordatela
fig,ax = plt.subplots()
for i in range(len(part_leaf)):
	
	box = part_leaf['box'].iloc[i]
	hull = ConvexHull(box)
	convex_hull_plot_2d(hull,ax=ax)
	#pp = Polygon(box, facecolor = 'none', edgecolor='b')
	#ax.add_patch(pp)
ax.set_xlim(-1.15,1.1)
ax.set_ylim(-1.1,1.1)
	#%%

for i in range(len(part)):
	fig,ax=plt.subplots()
	part['polytope'].iloc[i].plot(ax=ax,color='white', hatch=None, #alpha=0.3, 
									linestyle='solid', linewidth=0.8, edgecolor='black')
	ax.scatter(m[i][2][0],m[i][2][1])
	
ax.set_xlim(-1.15,1.1)
ax.set_ylim(-1.1,1.1)
	
	


