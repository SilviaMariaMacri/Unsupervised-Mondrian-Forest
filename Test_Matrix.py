from hypothesis import given,assume#,settings,HealthCheck
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

import numpy as np
from math import copysign
import pandas as pd


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
		
	it tests the correctness of cut_matrix and point_cut_distance of a 
	specific three sample dataset 
	'''
	
	assume(l!=0)
	X = np.array([[0,0],[l,0],[0,l]])
	data,cut_matrix,point_cut_distance = Matrix.cut_ensemble(X)
	
	magnitude_norm_vect = [abs(l/2),abs(l/2),0]
	norm_vect_0 = [copysign(1,l),0,-copysign(1,l)/np.sqrt(2)]
	norm_vect_1 = [0,copysign(1,l),copysign(1,l)/np.sqrt(2)]
	cut_matrix_true = {'magnitude_norm_vect':magnitude_norm_vect,'norm_vect_0':norm_vect_0,'norm_vect_1':norm_vect_1}
	cut_matrix_true = pd.DataFrame(cut_matrix_true)
	assert np.round(cut_matrix[['magnitude_norm_vect','norm_vect_0','norm_vect_1']],12-len(str(l))).equals(np.round(cut_matrix_true,12-len(str(l))))


	cut_index_0 = [-abs(l)/2,abs(l)/2,-abs(l)/2]
	cut_index_1 = [-abs(l)/2,-abs(l)/2,abs(l)/2]
	cut_index_2 = [0,-abs(l)/np.sqrt(2),abs(l)/np.sqrt(2)]
	point_index = [0,1,2]
	point_cut_distance_true = {'cut_index_0':cut_index_0,'cut_index_1':cut_index_1,'cut_index_2':cut_index_2,'point_index':point_index}
	point_cut_distance_true = pd.DataFrame(point_cut_distance_true)
	
	assert ((np.round(point_cut_distance,12-len(str(l)))).astype(float)).equals((np.round(point_cut_distance_true,12-len(str(l)))).astype(float))	
	
		
	
@given(l=st.integers(min_value=-1e+10,max_value=1e+10))
def test_cut_ensemble_four_samples(l):
	
	'''
	Matrix.cut_ensemble testing function:
		
	it tests the correctness of cut_matrix and point_cut_distance of a 
	specific four sample dataset
	'''
	
	assume(l!=0)
	X = np.array([[0,0],[l,0],[l,l],[0,l]])
	data,cut_matrix,point_cut_distance = Matrix.cut_ensemble(X)
	magnitude_norm_vect = [abs(l/2),abs(l/np.sqrt(2)),abs(l/2),abs(l/2),0,-abs(l/2)]
	norm_vect_0 = [copysign(1,l),copysign(1,l)/np.sqrt(2),0,0,-copysign(1,l)/np.sqrt(2),-copysign(1,l)]
	norm_vect_1 = [0,copysign(1,l)/np.sqrt(2),copysign(1,l),copysign(1,l),copysign(1,l)/np.sqrt(2),0]
	cut_matrix_true = {'magnitude_norm_vect':magnitude_norm_vect,'norm_vect_0':norm_vect_0,'norm_vect_1':norm_vect_1}
	cut_matrix_true = pd.DataFrame(cut_matrix_true)
	assert np.round(cut_matrix[['magnitude_norm_vect','norm_vect_0','norm_vect_1']],12-len(str(l))).equals(np.round(cut_matrix_true,12-len(str(l))))

	cut_index_0 = [-abs(l/2),abs(l/2),abs(l/2),-abs(l/2)]
	cut_index_1 = [-abs(l/np.sqrt(2)),0,abs(l/np.sqrt(2)),0]
	cut_index_2 = [-abs(l/2),-abs(l/2),abs(l/2),abs(l/2)]
	cut_index_3 = cut_index_2.copy()
	cut_index_4 = [0,-abs(l/np.sqrt(2)),0,abs(l/np.sqrt(2))]
	cut_index_5 = (-np.array(cut_index_0)).tolist().copy()
	point_index = [0,1,2,3]
	point_cut_distance_true = {'cut_index_0':cut_index_0,'cut_index_1':cut_index_1,'cut_index_2':cut_index_2,'cut_index_3':cut_index_3,'cut_index_4':cut_index_4,'cut_index_5':cut_index_5,'point_index':point_index}
	point_cut_distance_true = pd.DataFrame(point_cut_distance_true)
	assert ((np.round(point_cut_distance,12-len(str(l)))).astype(float)).equals((np.round(point_cut_distance_true,12-len(str(l)))).astype(float))	
	


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
		
	data,cut_matrix,point_cut_distance = Matrix.cut_ensemble(X)
	assert np.array_equal(np.array(data[data.columns[:-1]]),X)
	
	X_trasl = X+[x_trasl,y_trasl]
	data_trasl,cut_matrix_trasl,point_cut_distance_trasl = Matrix.cut_ensemble(X_trasl)
	assert cut_matrix[['norm_vect_0','norm_vect_1']].equals(cut_matrix_trasl[['norm_vect_0','norm_vect_1']])
	assert ((np.round(point_cut_distance_trasl,5)).astype(float)).equals((np.round(point_cut_distance,5)).astype(float)) 
	
	norm_vect =np.array(cut_matrix[['norm_vect_0','norm_vect_1']])
	magnitude_traslation = norm_vect*np.array([[x_trasl,y_trasl]])
	magnitude_traslation = np.sum(magnitude_traslation,axis=1)
	assert np.array_equal(np.round(np.array(cut_matrix_trasl[['magnitude_norm_vect']]).reshape((1,len(cut_matrix))),5),np.round(magnitude_traslation+np.array(cut_matrix[['magnitude_norm_vect']]).reshape((1,len(cut_matrix))),5))
					  




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
	
	data,cut_matrix,point_cut_distance = Matrix.cut_ensemble(X)
	
	theta = np.radians(theta_degree)
	rot_matrix = np.array([[np.cos(theta),-np.sin(theta)],
					       [np.sin(theta),np.cos(theta)]])
	X_rot = rot_matrix.dot(X.T).T
	data_rot,cut_matrix_rot,point_cut_distance_rot = Matrix.cut_ensemble(X_rot)
	assert np.round(cut_matrix['magnitude_norm_vect'].astype(float),7).equals(np.round(cut_matrix_rot['magnitude_norm_vect'].astype(float),7))
	assert np.round(point_cut_distance_rot.astype(float),7).equals(np.round(point_cut_distance.astype(float),7)) 
	
	norm_vect_init = np.array(cut_matrix[['norm_vect_0','norm_vect_1']])
	norm_vect_rot = np.array(cut_matrix_rot[['norm_vect_0','norm_vect_1']])
	norm_vect_init_rotated = rot_matrix.dot(norm_vect_init.T).T
	assert np.array_equal(np.round(norm_vect_rot,7),np.round(norm_vect_init_rotated,7))






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
	data,cut_matrix,point_cut_distance = Matrix.cut_ensemble(X)
	assert len(data) == len(X_unique)+1
	