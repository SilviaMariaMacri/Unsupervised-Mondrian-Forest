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
	p = {'id_number':part_number_list,'polytope':polytope_list}
	p = pd.DataFrame(p)
	
	neighbor_list = Merging.neighbors(p)
	neighbor_list_true = [[1,3],[0,2],[1],[0]]
	
	for i in range(len(p)):
		assert neighbor_list == neighbor_list_true
		
		
		
		
		
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
	part_space,part_data = Partitioning.partitioning(cut_ensemble,0,5,'min_corr',5)
	
	p = part_space.query('leaf==True').copy()
	p.index = np.arange(len(p))
	poly_number = p['id_number'].tolist()
	neighbor_list = Merging.neighbors(p)
	for i in range(len(p)):
		for j in neighbor_list[i]:
			assert poly_number[i] in neighbor_list[poly_number.index(j)]




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
	poly_number_input,neigh_poly_input,merged_poly_input,merg_data_input : 
		specific five polytope case, with fixed neighbor and merged polytope 
		list; each polytope contains one point
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
	
	poly_number_input = np.arange(5).tolist()
	neigh_poly_input = [[1],[0,2],[1,3],[2,4],[3]]
	merged_poly_input = [[5],[6],[7],[8],[9]]
	
	merg_data_input = [np.array([0]),np.array([1]),np.array([2]),np.array([3]),np.array([4])]
	
	part_to_remove = part_numbers[0]
	part_to_merge = part_numbers[1]
	assume(part_to_merge in neigh_poly_input[poly_number_input.index(part_to_remove)])
	
	poly_number,neigh_poly,merged_poly,merg_data = Merging.merge_two_polytopes(poly_number_input,neigh_poly_input,merged_poly_input,merg_data_input,part_to_remove,part_to_merge)
	
	#1
	assert part_to_remove not in poly_number
	#2
	assert len(poly_number) == len(poly_number_input)-1
	#3
	assert np.unique(poly_number).tolist() == poly_number
	
	#4
	neighbors_new = neigh_poly[poly_number.index(part_to_merge)]
	neighbors_part_to_remove = neigh_poly_input[poly_number_input.index(part_to_remove)]
	neighbors_part_to_merge = neigh_poly_input[poly_number_input.index(part_to_merge)]
	neighbors_new_true = list(set(neighbors_part_to_remove + neighbors_part_to_merge))
	
	neighbors_new_true.remove(part_to_remove)
	neighbors_new_true.remove(part_to_merge)
	assert sorted(neighbors_new) == neighbors_new_true
	
	#5
	merged_part_new = merged_poly[poly_number.index(part_to_merge)]
	merged_part_to_remove = merged_poly_input[poly_number_input.index(part_to_remove)]
	merged_part_to_merge = merged_poly_input[poly_number_input.index(part_to_merge)]
	assert merged_part_new == merged_part_to_merge + merged_part_to_remove + [part_to_remove]
	
	#6
	poly_number_input_not_changed = np.delete(poly_number_input,[poly_number_input.index(part_to_merge),poly_number_input.index(part_to_remove)])
	poly_number_not_changed = np.delete(poly_number,poly_number.index(part_to_merge))
	assert (poly_number_input_not_changed == poly_number_not_changed).all()

	neigh_poly_input_not_changed = np.delete(neigh_poly_input,[poly_number_input.index(part_to_merge),poly_number_input.index(part_to_remove)])
	neigh_poly_not_changed = np.delete(neigh_poly,poly_number.index(part_to_merge))	
	for i in range(len(poly_number_input_not_changed)):
		if part_to_remove in neigh_poly_input_not_changed[i]:
			list_to_change = sorted(neigh_poly_input_not_changed[i])
			list_to_change = [part_to_merge if value==part_to_remove else value for value in list_to_change]
			assert list_to_change == sorted(neigh_poly_not_changed[i])
		else:
			assert neigh_poly_input_not_changed[i] == sorted(neigh_poly_not_changed[i])
	
	#7
	assert len(merg_data) == len(merg_data_input)-1
	
	#8
	index_merged_part = poly_number.index(part_to_merge)
	index_part_to_merge = poly_number_input.index(part_to_merge)
	index_part_to_remove = poly_number_input.index(part_to_remove)
	merged_data = [merg_data_input[index_part_to_remove][0],merg_data_input[index_part_to_merge][0]] 
	assert (merg_data[index_merged_part] == merged_data).all()
	
	#9
	for i in range(len(merg_data)):
		if (i < part_to_remove) and (i != part_to_merge):
			assert merg_data[i] == merg_data_input[i]
		if (i > part_to_remove) and (i != part_to_merge):
			assert merg_data[i] == merg_data_input[i+1] 
	
	




##############################################################################
#   merge_single_data
##############################################################################

def test_merge_single_data():
	
	'''
	Merging.merge_single_data testing function: specific 2D case
	
	merge_single_data input:
	------------------------
	poly_number_input,neigh_poly_input,merged_poly_input : 
		specific five polytope case
	merg_data_input : list of dataset indexes associated to each polytope
	    two of them have one point and three of them have two points
	
	Tests:
	-----
	if the polytopes containing a single point have been merged
	'''
	
	poly_number_input = np.arange(5).tolist()
	neigh_poly_input = [[1],[0,2],[1,3],[2,4],[3]]
	merged_poly_input = [[5],[6],[7],[8],[9]]
	
	data = np.array([0.,0.1,1.,1.1,2,3.,3.1,4.])
	index = np.arange(len(data))
	data = np.vstack([data,index]).T
	merg_data_input = [np.array([0, 1]), np.array([2, 3]), np.array([4]), np.array([5, 6]), np.array([7])]

	poly_number,neigh_poly,merged_poly,merg_data = Merging.merge_single_data(poly_number_input,neigh_poly_input,merged_poly_input,merg_data_input,data)

	assert 2 not in poly_number
	assert 4 not in poly_number
	assert len(poly_number) == 3
