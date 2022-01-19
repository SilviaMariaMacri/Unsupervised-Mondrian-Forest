import numpy as np
from scipy.spatial.distance import cdist,pdist






def variance_metric(data1,data2):
	
	if (len(data1)>1) & (len(data2)>1):	
		pd1 = pdist(data1)
		pd2 = pdist(data2)
		data = np.vstack([data1,data2])
		pd = pdist(data)
		pd12 = np.hstack([pd1, pd2])
		var_ratio = np.var(pd)/np.var(pd12) 
		return var_ratio 
	else:
		s=np.nan
		return s



def centroid_metric(data1,data2):

	data12 = np.vstack([data1,data2])
	data_tot = [data1,data2,data12]
	centr=[[],[],[]]
	for j in range(3):
		for i in range(len(data12[0])):
			centr[j].append(np.mean(data_tot[j][:,i]))
	dist=[]
	for i in range(3):
		dist.append(cdist(data_tot[i],[centr[i]]))
	ratio = np.mean(dist[2])/np.mean(np.vstack([dist[0],dist[1]]))	
	difference = np.mean(dist[2]) - np.mean(np.vstack([dist[0],dist[1]]))

	return ratio,difference	  	  




def min_dist_metric(data1,data2):
	
	pd1 = cdist(data1,data1)
	pd2 = cdist(data2,data2)
				
	min1 = np.min(np.where(pd1!= 0, pd1, np.inf),axis=0)
	min2 = np.min(np.where(pd2!= 0, pd2, np.inf),axis=0)
				
	mean1 = np.mean(min1)
	mean2 = np.mean(min2)	
	
	min_tot = np.hstack([min1,min2])
	if np.inf in min_tot:
		min_tot = list(min_tot)
		min_tot.remove(np.inf)
	media = np.mean(min_tot)
			
	dist = cdist(data1,data2)
	min_dist_between_subspaces = dist.min()
	ind = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
	#da ind ricavo A e B
			
	dist_point1 = pd1[ind[0]]
	dist_point1 = np.where(dist_point1!= 0, dist_point1, np.inf)
	ind1 = np.unravel_index(np.argmin(dist_point1, axis=None), dist_point1.shape)
	ind1 = ind1[0] 
	dist1 = dist[ind1,:]
	min_dist1 = np.min(dist1)
			
	dist_point2 = pd2[ind[1]]
	dist_point2 = np.where(dist_point2!= 0, dist_point2, np.inf)
	ind2 = np.unravel_index(np.argmin(dist_point2, axis=None), dist_point2.shape)
	ind2 = ind2[0] #nuovo punto in data2
	dist2 = dist[:,ind2]
	min_dist2 = np.min(dist2)

	return min_dist_between_subspaces,media,min_dist1,min_dist2,mean1,mean2





def compute_metric(metric,data1,data2):
	
	data1 = data1.drop('index',axis=1)
	data2 = data2.drop('index',axis=1)
	data1 = np.array(data1)
	data2 = np.array(data2)
	
	if metric == 'variance':	
		var_ratio = variance_metric(data1,data2)
		metric_value = var_ratio
			
	if metric == 'centroid_diff':	
		ratio,difference = centroid_metric(data1, data2)
		metric_value = difference

	if metric == 'centroid_ratio':	
		ratio,difference = centroid_metric(data1, data2)
		metric_value = ratio
			
	if metric == 'min':
		min_dist_between_subspaces,media,min_dist1,min_dist2,mean1,mean2 = min_dist_metric(data1,data2)
		diff = abs(min_dist_between_subspaces - media)
		metric_value = diff
			
	if metric == 'min_corr':	
		min_dist_between_subspaces,media,min_dist1,min_dist2,mean1,mean2 = min_dist_metric(data1,data2)
		if len(data1) == 1:
			diff = abs(min_dist_between_subspaces - media) + min_dist2
		if len(data2) == 1:
			diff = abs(min_dist_between_subspaces - media) + min_dist1 
		else:
			diff = abs(min_dist_between_subspaces - media) + min_dist1 + min_dist2
		metric_value = diff

	return metric_value

