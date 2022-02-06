'''
#neighbors
#merge_two_polytopes
#merge_single_data
polytope_similarity
polytope_similarity_update
NO merging
'''

from hypothesis import given,assume,settings#,HealthCheck
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

import polytope as pc
import numpy as np
import pandas as pd

import Matrix
import Partitioning
import Merging




##############################################################################
#   neighbors
##############################################################################


def test_neighbors_2D_case():
	
	'''
	Merging.neighbors testing function: specific 2D case
	
	given four adjacent rectangles, it test if the calculated neighbors
	of each polygon are the correct ones
	'''

	A1 = np.array([[-1,0],[1,0],[0,-1],[0,1]])
	b1 = np.array([-5,10,0,10])
	p1 = pc.Polytope(A1,b1)
	
	A2 = np.array([[-1,0],[1,0],[0,-1],[0,1]])
	b2 = np.array([5,5,-5,10])
	p2 = pc.Polytope(A2,b2)
	
	A3 = np.array([[-1,0],[1,0],[0,-1],[0,1]])
	b3 = np.array([5,0,5,5])
	p3 = pc.Polytope(A3,b3)
	
	A4 = np.array([[-1,0],[1,0],[0,-1],[0,1]])
	b4 = np.array([-10,15,5,0])
	p4 = pc.Polytope(A4,b4)
	
	polytope_list = [p1,p2,p3,p4]
	part_number_list = np.arange(len(polytope_list))
	p = {'part_number':part_number_list,'polytope':polytope_list}
	p = pd.DataFrame(p)
	
	p = Merging.neighbors(p)
	neighbor_list = [[1,3],[0,2],[1],[0]]
	
	for i in range(len(p)):
		assert list(p['neighbors'].iloc[i]) == neighbor_list[i]
		
		
		
		
		
@given(X = arrays(dtype=np.int64,
				  shape=st.tuples(st.integers(min_value=3,max_value=20),st.integers(min_value=2,max_value=10)),
				  elements=st.integers(min_value=-50,max_value=50),
				  unique=True))
@settings(deadline=None)
def test_neighbors_2(X):
	
	'''
	Merging.neighbors testing function: 
	
	Input:
	-----
	X : dataset of variable length and dimension
		it is generated in order to perform its hierarchical partitioning 
		through the Partitioning.partitioning function; the final result of 
		the partitioning phase is used as input of Merging.neighbors 
	
	Tests:
	-----
	if each polytope is on the neighbor list of each neighbor
	'''

	cut_ensemble = Matrix.cut_ensemble(X)
	part,m = Partitioning.partitioning(cut_ensemble,0,5,'min_corr',5)
	
	p = part.query('leaf==True').copy()
	p.index = np.arange(len(p))
	p = Merging.neighbors(p)
	for i in range(len(p)):
		for j in p['neighbors'].iloc[i]:
			assert p['part_number'].iloc[i] in list(p[p['part_number']==j]['neighbors'])[0]





##############################################################################
#   merge_two_polytopes
##############################################################################


@given(part_numbers = st.tuples(st.integers(min_value=0,max_value=4),st.integers(min_value=0,max_value=4)))
def test_merge_two_polytopes(part_numbers):
	
	'''
	VERIFICA SE TUTTI QUESTI TEST HANNO SENSO
	
	Merging.merge_two_polytopes testing function:
	specific case with different merging
	
	merge_two_polytope input:
	------------------------
	p_init,m_init : specific five polytope case, with fixed neighbor and 
		merged polytope list; each polytope contains one point
	part_numbers : variable tuple of two integers
		the two numbers correspond to the input parameters part_to_remove
		and part_to_merge
		
	Tests:
	-----
	1. if the removed polytope is not in the output list of polytopes
	2. if the number of polytopes of the output list is equal to the number of
	   the input list minus one
	3. if no polytope of the output list is repeated
	4. if the neighbor list of the merged polytope is the correct one
	5. if the merged polytope list of the merged polytope is the correct one
	6. if the informations about the polytopes that are not involved in the 
	   merging procedure have not been changed or have been changed in the 
	   correct way
	7. if the number of dataframes storing the samples of the output list 
	   is equal to the number of the input list minus one
	8. if the datasets have been merged in the correct way
	9. if the datasets not involved in the merging procedure have not been changed         
	'''
	
	assume(part_numbers[0] != part_numbers[1])
	
	part_number = np.arange(5).tolist()
	neighbors = [[1],[0,2],[1,3],[2,4],[3]]
	merged_part = [[5],[6],[7],[8],[9]]
	
	p_init = {'part_number':part_number,'neighbors':neighbors,'merged_part':merged_part}
	p_init = pd.DataFrame(p_init)
	
	m_leaf_init = []
	for i in range(5):
		m_leaf_i = {'x':[i]}
		m_leaf_i = pd.DataFrame(m_leaf_i)
		m_leaf_init.append(m_leaf_i)
	
	part_to_remove = part_numbers[0]
	part_to_merge = part_numbers[1]
	assume(part_to_merge in list(p_init[p_init['part_number']==part_to_remove]['neighbors'])[0])
	
	p,m_leaf = Merging.merge_two_polytopes(p_init,m_leaf_init,part_to_remove,part_to_merge)
	
	#1
	assert part_to_remove not in list(p['part_number'])
	#2
	assert len(p) == len(p_init)-1
	#3
	assert p['part_number'].unique().tolist() == list(p['part_number'])
	
	#4
	neighbors_new = list(p[p['part_number']==part_to_merge]['neighbors'])[0]
	neighbors_part_to_remove = list(p_init[p_init['part_number']==part_to_remove]['neighbors'])[0]
	neighbors_part_to_merge = list(p_init[p_init['part_number']==part_to_merge]['neighbors'])[0]
	neighbors_new_true = list(set(neighbors_part_to_remove + neighbors_part_to_merge))
	neighbors_new_true.remove(part_to_remove)
	neighbors_new_true.remove(part_to_merge)
	assert sorted(neighbors_new) == neighbors_new_true
	
	#5
	merged_part_new = list(p[p['part_number']==part_to_merge]['merged_part'])[0]
	merged_part_to_remove = list(p_init[p_init['part_number']==part_to_remove]['merged_part'])[0]
	merged_part_to_merge = list(p_init[p_init['part_number']==part_to_merge]['merged_part'])[0]
	assert merged_part_new == merged_part_to_merge + merged_part_to_remove + [part_to_remove]
	
	#6
	p_init_not_changed = p_init.query('part_number!='+str(part_to_merge)+' and part_number!='+str(part_to_remove)).copy()
	p_init_not_changed.index = np.arange(len(p_init_not_changed))
	p_not_changed = p.query('part_number!='+str(part_to_merge)).copy()
	p_not_changed.index = np.arange(len(p_not_changed))
	assert p_init_not_changed[['part_number','merged_part']].equals(p_not_changed[['part_number','merged_part']])
	
	for i in range(len(p_init_not_changed)):
		if part_to_remove in list(p_init_not_changed['neighbors'].iloc[i]):
			list_to_change = sorted(list(p_init_not_changed['neighbors'].iloc[i]))
			list_to_change = [part_to_merge if value==part_to_remove else value for value in list_to_change]
			assert list_to_change == sorted(list(p_not_changed['neighbors'].iloc[i]))
		else:
			assert list(p_init_not_changed['neighbors'].iloc[i]) == sorted(list(p_not_changed['neighbors'].iloc[i]))
	
	#7
	assert len(m_leaf) == len(m_leaf_init)-1
	index_merged_part = p[p['part_number']==part_to_merge].index[0]
	index_part_to_merge = p_init[p_init['part_number']==part_to_merge].index[0]
	index_part_to_remove = p_init[p_init['part_number']==part_to_remove].index[0]
	
	#8
	merged_data = pd.concat([m_leaf_init[index_part_to_merge],m_leaf_init[index_part_to_remove]])
	merged_data.index = np.arange(len(merged_data))
	assert m_leaf[index_merged_part].equals(merged_data)
	
	#9
	for i in range(len(m_leaf)):
		if (i < part_to_remove) and (i != part_to_merge):
			assert m_leaf[i].equals(m_leaf_init[i]) 
		if (i > part_to_remove) and (i != part_to_merge):
			assert m_leaf[i].equals(m_leaf_init[i+1]) 
	




##############################################################################
#   merge_single_data
##############################################################################

def test_merge_single_data():
	
	'''
	Merging.merge_single_data testing function: specific 2D case
	
	merge_single_data input:
	------------------------
	p_init : specific five polytope case
	m_init : list of datasets associated to each polytope
		two of them have one point and three of them have two points
	
	Tests:
	-----
	if the polytopes containing a single point have been merged
	'''
	
	part_number = np.arange(5).tolist()
	neighbors = [[1],[0,2],[1,3],[2,4],[3]]
	merged_part = [[5],[6],[7],[8],[9]]
	
	p_init = {'part_number':part_number,'neighbors':neighbors,'merged_part':merged_part}
	p_init = pd.DataFrame(p_init)
	
	m_leaf_init = []
	index = 0
	for i in range(5):
		if i in [0,1,3]:
			m_leaf_i = {'x':[i,i+0.1]}
			m_leaf_i = pd.DataFrame(m_leaf_i)
			m_leaf_i['index'] = [index,index+1]
			m_leaf_init.append(m_leaf_i)
			index += 2
		else:
			m_leaf_i = {'x':[i]}
			m_leaf_i = pd.DataFrame(m_leaf_i)
			m_leaf_i['index'] = [index]
			m_leaf_init.append(m_leaf_i)
			index += 1			

	p,m_leaf = Merging.merge_single_data(p_init,m_leaf_init)

	assert 2 not in list(p['part_number'])
	assert 4 not in list(p['part_number'])
	assert len(p) == 3
	

