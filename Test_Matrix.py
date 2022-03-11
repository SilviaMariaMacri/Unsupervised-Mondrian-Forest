from hypothesis import given,assume#,settings,HealthCheck
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

import numpy as np
from math import copysign

import Matrix






def test_find_equal_columns():
	
	matrix = np.array([[1,1,1,1,0,1],[1,1,0,0,0,0],[1,1,1,1,0,0]])
	equal_columns,equal_column_index = Matrix.find_equal_columns(matrix)
	
	assert equal_columns.tolist() == [0,1,4,2,3,5]
	assert equal_column_index.tolist() == [0,0,0,1,1,2]





@given(l=st.integers(min_value=-1e+10,max_value=1e+10))
def test_cut_ensemble_three_samples(l):
	
	'''
	Matrix.cut_ensemble testing function:
		
	it tests the correctness of cut_matrix and distance_matrix of a 
	specific three sample dataset 
	'''
	
	assume(l!=0)
	X = np.array([[0,0],[l,0],[0,l]])
	data,cut_matrix,distance_matrix = Matrix.cut_ensemble(X)
	
	magnitude_norm_vect = [abs(l/2),abs(l/2),0]
	norm_vect_0 = [copysign(1,l),0,-copysign(1,l)/np.sqrt(2)]
	norm_vect_1 = [0,copysign(1,l),copysign(1,l)/np.sqrt(2)]
	cut_matrix_true = np.vstack([magnitude_norm_vect,norm_vect_0,norm_vect_1]).T
	assert (np.round(cut_matrix[:,-3:],12-len(str(l))) == np.round(cut_matrix_true,12-len(str(l)))).all()

	cut_index_0 = [-abs(l)/2,abs(l)/2,-abs(l)/2]
	cut_index_1 = [-abs(l)/2,-abs(l)/2,abs(l)/2]
	cut_index_2 = [0,-abs(l)/np.sqrt(2),abs(l)/np.sqrt(2)]
	point_index = [0,1,2]
	distance_matrix_true = np.vstack([cut_index_0,cut_index_1,cut_index_2,point_index]).T
	assert (((np.round(distance_matrix,12-len(str(l)))).astype(float)) == (np.round(distance_matrix_true,12-len(str(l)))).astype(float)).all()	
	
		
	
@given(l=st.integers(min_value=-1e+10,max_value=1e+10))
def test_cut_ensemble_four_samples(l):
	
	'''
	Matrix.cut_ensemble testing function:
		
	it tests the correctness of cut_matrix and point_cut_distance of a 
	specific four sample dataset
	'''
	
	assume(l!=0)
	X = np.array([[0,0],[l,0],[l,l],[0,l]])
	data,cut_matrix,distance_matrix = Matrix.cut_ensemble(X)
	magnitude_norm_vect = [abs(l/2),abs(l/np.sqrt(2)),abs(l/2),abs(l/2),0,-abs(l/2)]
	norm_vect_0 = [copysign(1,l),copysign(1,l)/np.sqrt(2),0,0,-copysign(1,l)/np.sqrt(2),-copysign(1,l)]
	norm_vect_1 = [0,copysign(1,l)/np.sqrt(2),copysign(1,l),copysign(1,l),copysign(1,l)/np.sqrt(2),0]
	cut_matrix_true = np.vstack([magnitude_norm_vect,norm_vect_0,norm_vect_1]).T
	assert (np.round(cut_matrix[:,-3:],12-len(str(l))) == np.round(cut_matrix_true,12-len(str(l)))).all()

	cut_index_0 = [-abs(l/2),abs(l/2),abs(l/2),-abs(l/2)]
	cut_index_1 = [-abs(l/np.sqrt(2)),0,abs(l/np.sqrt(2)),0]
	cut_index_2 = [-abs(l/2),-abs(l/2),abs(l/2),abs(l/2)]
	cut_index_3 = cut_index_2.copy()
	cut_index_4 = [0,-abs(l/np.sqrt(2)),0,abs(l/np.sqrt(2))]
	cut_index_5 = (-np.array(cut_index_0)).tolist().copy()
	point_index = [0,1,2,3]
	distance_matrix_true = np.vstack([cut_index_0,cut_index_1,cut_index_2,cut_index_3,cut_index_4,cut_index_5,point_index]).T
	assert (((np.round(distance_matrix,12-len(str(l)))).astype(float)) == (np.round(distance_matrix_true,12-len(str(l)))).astype(float)).all()	
	


@given(X = arrays(dtype=np.int64,
		          shape=st.tuples(st.integers(min_value=2,max_value=20),
				  st.integers(min_value=2,max_value=2)),
				  elements=st.integers(min_value=-1e+2,max_value=1e+2),
				  unique=True),
	   x_trasl = st.integers(min_value=-1e+2,max_value=1e+2),
	   y_trasl = st.integers(min_value=-1e+2,max_value=1e+2))
def test_cut_ensemble_traslation(X,x_trasl,y_trasl):
	
	'''
	Matrix.cut_ensemble testing function:
	
	Input:
	-----	 
	X : array representing a 2 dimensional dataset of maximum 20 samples
		and integer elements
	x_trasl,y_trasl : integers coordinates of the traslation vector
	
	Tests:
	-----
	- if the output indexed data are the same of the input data		
	- if the hyperplane directions are traslation-invariant
	- if the point-hyperplane distances are traslation-invariant
	- if the origin-hyperplane distances of the traslated dataset are equal
	  to the sum of the origin-hyperplane distances of the initial dataset
	  and the length of the projection of the traslation vector on the
	  hyperplane vector
	'''
		
	data,cut_matrix,distance_matrix = Matrix.cut_ensemble(X)
	assert (data[:,0:-1] == X).all()
	
	X_trasl = X+[x_trasl,y_trasl]
	data_trasl,cut_matrix_trasl,distance_matrix_trasl = Matrix.cut_ensemble(X_trasl)
	
	assert (cut_matrix[:,-2:] == cut_matrix_trasl[:,-2:]).all()
	assert ((np.round(distance_matrix_trasl,5)).astype(float) == (np.round(distance_matrix,5)).astype(float)).all() 
	
	norm_vect = cut_matrix[:,-2:].copy()
	magnitude_traslation = norm_vect*np.array([[x_trasl,y_trasl]])
	magnitude_traslation = np.sum(magnitude_traslation,axis=1)
	assert (np.round(cut_matrix_trasl[:,4],5) == np.round(magnitude_traslation+cut_matrix[:,4],5)).all()		  




@given(X = arrays(dtype=np.int64,
		          shape=st.tuples(st.integers(min_value=2,max_value=20),st.integers(min_value=2,max_value=2)),
				  elements=st.integers(min_value=-100,max_value=100),
				  unique=True),
	   theta_degree = st.integers(min_value=0,max_value=360))#,allow_nan=False
def test_cut_ensemble_rotation(X,theta_degree):
	
	'''
	Matrix.cut_ensemble testing function:
	
	Input:
	-----
	X : array representing a 2 dimensional dataset of maximum 20 samples
		and integer elements
	theta_degree : integers value of the rotation angle
	
	Tests:
	-----
	- if the origin-hyperplane distances are rotation-invariant
	- if the point-hyperplane distances are rotation-invariant
	- if the hyperplane directions of the rotated dataset are equal to the 
	  rotated hyperplane directions of the initial dataset
	'''
	
	data,cut_matrix,distance_matrix = Matrix.cut_ensemble(X)
	
	theta = np.radians(theta_degree)
	rot_matrix = np.array([[np.cos(theta),-np.sin(theta)],
					       [np.sin(theta),np.cos(theta)]])
	X_rot = rot_matrix.dot(X.T).T
	data_rot,cut_matrix_rot,distance_matrix_rot = Matrix.cut_ensemble(X_rot)
	
	assert (np.round(cut_matrix[:,4].astype(float),7) == np.round(cut_matrix_rot[:,4].astype(float),7)).all()
	assert (np.round(distance_matrix_rot.astype(float),7) == np.round(distance_matrix.astype(float),7)).all() 
	
	norm_vect_init = cut_matrix[:,-2:]
	norm_vect_rot = cut_matrix_rot[:,-2:]
	norm_vect_init_rotated = rot_matrix.dot(norm_vect_init.T).T
	assert (np.round(norm_vect_rot,7) == np.round(norm_vect_init_rotated,7)).all()



@given(len_X_unique = st.integers(min_value=2,max_value=20),
	   len_X_duplicates = st.integers(min_value=1,max_value=50))
def test_cut_ensemble_duplicates(len_X_unique,len_X_duplicates):
	
	'''
	Matrix.cut_ensemble testing function:
		
	Tests the reduction of the input dataset in case of duplicate samples
	'''
	
	X_unique = np.arange(len_X_unique).reshape((len_X_unique,1))
	X_duplicates = 20*np.ones((len_X_duplicates,1))
	X = np.vstack([X_unique,X_duplicates])
	data,cut_matrix,distance_matrix = Matrix.cut_ensemble(X)
	assert len(data) == len(X_unique)+1	