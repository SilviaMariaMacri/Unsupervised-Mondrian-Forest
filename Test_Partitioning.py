from hypothesis import given,settings
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

import polytope as pc
import numpy as np
import pandas as pd
from numpy.linalg import norm

import Matrix
import Partitioning

'''
#data_splitting
NO cut_choice   
#space_splitting   
#data_assignment
#recursive_process
NO partitioning
'''


##############################################################################
#   data_splitting
##############################################################################

def test_data_splitting():
	
	'''
	Partitioning.data_splitting testing function:
		
	given a hyperplane separating a 2D dataset into two subsets of 11 and 12 
	samples, it tests if the two output subsets of points are the correct ones
	'''
	
	X_pos = np.random.rand(10,2)
	X_neg = -np.random.rand(11,2)
	X_init = np.array([[1,0],[-1,0]])
	X = np.vstack([X_init,X_pos,X_neg])

	data,cut_matrix,point_cut_distance = Matrix.cut_ensemble(X)
	
	cut_index = 0
	data_pos,data_neg = Partitioning.data_splitting(data,cut_index,point_cut_distance)
	
	assert ((len(data_pos) == 11) and (len(data_neg) == 12)) or ((len(data_pos) == 12) and (len(data_neg) == 11))
	assert (np.array(data_pos[['0','1']])[1:].tolist()== X_pos.tolist()) or (np.array(data_pos[['0','1']])[1:].tolist()== X_neg.tolist())
	assert (np.array(data_neg[['0','1']])[1:].tolist()== X_neg.tolist()) or (np.array(data_neg[['0','1']])[1:].tolist()== X_pos.tolist())





##############################################################################
#   space_splitting
##############################################################################

@given(l=st.integers(min_value=1,max_value=1e+3))
def test_space_splitting_1(l):
	
	'''
	Partitioning.space_splitting testing function: specific 2D case
	
	space_splitting input :
	----------------------
	p : 2D square, centered in (0,0)
	hyperplane : first axis
	
	Tests:
	-----
	- if the two output polytopes have equal volume
	- if the matrices A and b that define each polytope are the correct ones
	'''
	
	A = np.array([[1,0],[-1,0],[0,1],[0,-1]])
	b = l*np.ones(4)
	p = pc.Polytope(A,b)
	hyperplane_direction = np.array([1,0])
	hyperplane_distance = 0
	p1,p2 = Partitioning.space_splitting(p,hyperplane_direction,hyperplane_distance)
	
	assert p1.volume == p2.volume
	assert p.A.tolist() == p1.A[[3,0,1,2]].tolist()
	assert p.A.tolist() == p2.A[[0,3,1,2]].tolist()
	assert np.round(p1.b,0).tolist() == np.round(np.hstack([p.b[0:-1],np.array([0])]),0).tolist()
	assert np.round(p2.b,0).tolist() == np.round(np.hstack([p.b[0:-1],np.array([0])]),0).tolist()
	
	
	
	


	
@given(hyperplane_coordinates = arrays(dtype=float,
		          shape=st.integers(min_value=2,max_value=20),
				  elements=st.floats(min_value=-1,max_value=1,allow_nan=False),
				  unique=True),
	   hyperplane_distance = st.integers(min_value=1,max_value=9))
def test_space_splitting_4(hyperplane_coordinates,hyperplane_distance):
	
	'''
	Partitioning.space_splitting testing function:
	
	space_splitting input :
	----------------------
	p : axis-aligned box with variable dimension 
	hyperplane : hyperplane with variable direction 
	
	Tests:
	-----
	- if the union of the two output polytopes is equal to the input polytope
	- if the intersection of the output polytopes is null
	'''
	
	n_dim = len(hyperplane_coordinates)
	#assume(sum(hyperplane_coordinates[0]) != 0)
	#assume(np.array_equal(hyperplane_coordinates[0],np.zeros(n_dim)))
	
	A = []
	for i in range(n_dim):
		A_i1 = list(np.zeros(n_dim))
		A_i1[i] = 1.
		A_i2 = list(np.zeros(n_dim))
		A_i2[i] = -1.
		A.append(A_i1)
		A.append(A_i2)
	A = np.array(A)
	b = 10*np.ones(2*n_dim)
	p = pc.Polytope(A,b)
	
	normalization = norm(hyperplane_coordinates)
	hyperplane_direction = hyperplane_coordinates/normalization
	p1,p2 = Partitioning.space_splitting(p,hyperplane_direction,hyperplane_distance)
	
	assert p1.union(p2) == p
	assert p1.intersect(p2).A.tolist() == []
	assert p1.intersect(p2).b.tolist() == []






def test_space_splitting_limit_case():
		
	'''
	Partitioning.space_splitting testing function: 
	limit case of intersecting hyperplane coincident with a polytope face
	
	space_splitting input :
	----------------------
	p : 3D cube, centered in (0,0,0)
	hyperplane : a cube face
	
	Tests:
	-----
	- if the first output polytope is equal to the input one
	- if the second output polytope is zero dimensioned
	'''

	A = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
	b = 10*np.ones(6)
	p = pc.Polytope(A,b)
	hyperplane_direction = np.array([1,0,0])
	hyperplane_distance = 10	
	
	p1,p2 = Partitioning.space_splitting(p,hyperplane_direction,hyperplane_distance)
	assert p1 == p
	assert p2.A.tolist() == []
	assert p2.b.tolist() == []	






##############################################################################
#   data_assignment
##############################################################################

@given(data_pos = arrays(dtype=float,
						   shape=st.tuples(st.integers(min_value=2,max_value=30),st.integers(min_value=3,max_value=3)),
						   elements=st.floats(min_value=1,max_value=9),
						   unique=True),
	   data_neg = arrays(dtype=float,
						   shape=st.tuples(st.integers(min_value=2,max_value=30),st.integers(min_value=3,max_value=3)),
						   elements=st.floats(min_value=-9,max_value=-1),
						   unique=True))
def test_data_assignment(data_pos,data_neg):
	
	'''
	Partitioning.data_assignment testing function: 
	
	data_assignment input :
	----------------------
	p1,p2 : two 3D cubes 
	data_pos,data_neg : two sets of points, each one contained in one  
	                    of the two polytopes
	
	Tests:
	-----
	- if the sets are correctly assigned, independently from the order of 
	  data_pos and data_neg as input parameters
	'''
	
	A1 = np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
	b1 = np.array([0,10,0,10,0,10])
	p1 = pc.Polytope(A1,b1)
	
	A2 = np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
	b2 = np.array([10,0,10,0,10,0])
	p2 = pc.Polytope(A2,b2)
	
	data_pos = pd.DataFrame(data_pos)
	data_pos['index'] = np.arange(len(data_pos))
	data_neg = pd.DataFrame(data_neg)
	data_neg['index'] = np.arange(len(data_neg))
	
	data1a,data2a = Partitioning.data_assignment(p1,p2,data_pos,data_neg)
	assert data_pos.equals(data1a)
	assert data_neg.equals(data2a)
	
	data1b,data2b = Partitioning.data_assignment(p1,p2,data_neg,data_pos)
	assert data_pos.equals(data1b)
	assert data_neg.equals(data2b)
	
	
	 
	
def test_data_assignment_limit_case():
	
	'''
	Partitioning.data_assignment testing function: 
	limit case with points lying on the adjacent face of the two polytopes
	
	data_assignment input :
	----------------------
	p1,p2 : two 2D squares 
	data_pos : dataset consisting of two points
		the first point lies on the common side of the squares, while the
		second one belongs to p1
	data_neg : dataset consisting of two points
		the first point lies on the common side of the squares, while the
		second one belongs to p2
		
	Tests:
	-----
	- if the sets are correctly assigned, independently from the order of 
	  data_pos and data_neg as input parameters
	'''
	
	A1 = np.array([[-1,0],[1,0],[0,-1],[0,1]])
	b1 = np.array([0,10,0,10])
	p1 = pc.Polytope(A1,b1)
	
	A2 = np.array([[-1,0],[1,0],[0,-1],[0,1]])
	b2 = np.array([10,0,0,10])
	p2 = pc.Polytope(A2,b2)
	
	limit_point_pos = np.array([[10,5]])
	limit_point_neg = np.array([[-10,5]])
	limit_point_middle = np.array([[0,5]])
	
	data_pos = np.vstack([limit_point_middle,limit_point_pos])
	data_pos = pd.DataFrame(data_pos)
	data_pos['index'] = np.arange(len(data_pos))
	
	data_neg = np.vstack([limit_point_middle,limit_point_neg])
	data_neg = pd.DataFrame(data_neg)
	data_neg['index'] = np.arange(len(data_neg))
	
	data1a,data2a = Partitioning.data_assignment(p1,p2,data_pos,data_neg)
	assert data_pos.equals(data1a)
	assert data_neg.equals(data2a)
	
	data1b,data2b = Partitioning.data_assignment(p1,p2,data_neg,data_pos)
	assert data_pos.equals(data1b)
	assert data_neg.equals(data2b)






##############################################################################
#   recursive_process
##############################################################################

@given(X = arrays(dtype=np.int64,
				  shape=st.tuples(st.integers(min_value=3,max_value=20),st.integers(min_value=2,max_value=10)),
				  elements=st.integers(min_value=-50,max_value=50),
				  unique=True),
	  metric_index = st.integers(min_value=0,max_value=4))
@settings(deadline=None)
def test_recursive_process(X,metric_index):
	
	'''
	Partitioning.recursive_process testing function: 
	
	Input :
	----------------------
	X : dataset of variable length and dimension
	metric_index : allows to choose different metrics in phase of splitting 
		procedure testing
		
	Tests:
	-----
	- if the two subsets of points are completely contained in the 
	  corresponding polytopes
	'''
	
	data,cut_matrix,point_cut_distance = Matrix.cut_ensemble(X)
	
	# initial space
	n_d = len(data.columns)-1
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
	
	metric = ['variance','centroid_ratio','centroid_diff','min','min_corr']
	rec_process_result = Partitioning.recursive_process(p,data,cut_matrix,point_cut_distance,0,1e+5,metric[metric_index],5)
	
	if isinstance(rec_process_result, list):
		p1,data1,p2,data2,t0 = rec_process_result
		for i in range(len(data1)):
			point_data1 = data1.iloc[i].copy()
			point_data1 = list(point_data1[0:-1])
			assert point_data1 in p1
		for i in range(len(data2)):
			point_data2 = data2.iloc[i].copy()
			point_data2 = list(point_data2[0:-1])
			assert point_data2 in p2




def test_recursive_process_2samples():
	
	'''
	Partitioning.recursive_process testing function: 
	
	It tests that the splitting procedure is not performed in case of 
	input dataset consisting of two points
	'''
	
	X = np.array([[1,1],[3,4]])
	data,cut_matrix,point_cut_distance = Matrix.cut_ensemble(X)
	
	# initial space
	n_d = len(data.columns)-1
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
	
	rec_process_result = Partitioning.recursive_process(p,data,cut_matrix,point_cut_distance,0,1e+5,'min_corr',5)
	
	assert rec_process_result == None		




def test_recursive_process_lifetime():
	
	'''
	Partitioning.recursive_process testing function: 
	
	It tests that the splitting procedure is not performed in case of 
	initial time higher than the lifetime
	'''
	
	t0 = 5
	lifetime = 3
	
	X = np.array([[1,1],[3,4],[6,7]])
	data,cut_matrix,point_cut_distance = Matrix.cut_ensemble(X)
	
	# initial space
	n_d = len(data.columns)-1
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
	
	rec_process_result = Partitioning.recursive_process(p,data,cut_matrix,point_cut_distance,t0,lifetime,'min_corr',5)
	
	assert rec_process_result == None						