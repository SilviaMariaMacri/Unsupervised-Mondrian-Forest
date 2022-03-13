from hypothesis import given,settings,assume,HealthCheck
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

import numpy as np
import pandas as pd

import Metrics




@given(data1 = arrays(dtype=np.int64,
				  shape=st.tuples(st.integers(min_value=3,max_value=20),st.integers(min_value=5,max_value=5)),
				  elements=st.integers(min_value=1,max_value=50),
				  unique=True),
	   data2 = arrays(dtype=np.int64,
				  shape=st.tuples(st.integers(min_value=3,max_value=20),st.integers(min_value=5,max_value=5)),
				  elements=st.integers(min_value=-50,max_value=-1),
				  unique=True),
	   metric_index = st.integers(min_value=0,max_value=4),
	   trasl_vector = arrays(dtype=np.int64,
		          shape=5,
				  elements=st.integers(min_value=-100,max_value=100),
				  unique=True))
@settings(deadline=None,suppress_health_check=[HealthCheck.filter_too_much])
def test_compute_metric_traslation(data1,data2,metric_index,trasl_vector):
	
	assume(np.array_equal(data1,data2) == False)
	
	data1_trasl = data1+trasl_vector
	data2_trasl = data2+trasl_vector

	metric_list = ['variance','centroid_ratio','centroid_diff','min','min_corr']
	metric = metric_list[metric_index]
	
	metric_value = Metrics.compute_metric(metric,data1,data2)
	metric_value_trasl = Metrics.compute_metric(metric,data1_trasl,data2_trasl)
	
	assert np.round(metric_value,8) == np.round(metric_value_trasl,8)
	





@given(data1 = arrays(dtype=np.int64,
				  shape=st.tuples(st.integers(min_value=3,max_value=20),st.integers(min_value=2,max_value=2)),
				  elements=st.integers(min_value=1,max_value=50),
				  unique=True),
	   data2 = arrays(dtype=np.int64,
				  shape=st.tuples(st.integers(min_value=3,max_value=20),st.integers(min_value=2,max_value=2)),
				  elements=st.integers(min_value=-50,max_value=-1),
				  unique=True),
	   metric_index = st.integers(min_value=0,max_value=4),
	   theta_degree = st.integers(min_value=10,max_value=350))
@settings(deadline=None)
def test_compute_metric_rotation(data1,data2,metric_index,theta_degree):
	
	assume(np.array_equal(data1,data2) == False)
	
	theta = np.radians(theta_degree)
	rot_matrix = np.array([[np.cos(theta),-np.sin(theta)],
					       [np.sin(theta),np.cos(theta)]])
	data1_rot = rot_matrix.dot(data1.T).T
	data2_rot = rot_matrix.dot(data2.T).T

	metric_list = ['variance','centroid_ratio','centroid_diff','min','min_corr']
	metric = metric_list[metric_index]
	
	metric_value = Metrics.compute_metric(metric,data1,data2)
	metric_value_rot = Metrics.compute_metric(metric,data1_rot,data2_rot)
	
	assert np.round(metric_value,8) == np.round(metric_value_rot,8)

