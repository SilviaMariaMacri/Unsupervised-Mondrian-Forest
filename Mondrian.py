import pandas as pd
import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score
from itertools import combinations
import polytope as pc
import json

import Matrix
import Partitioning
import Merging




def mondrian_tree(X,t0,lifetime,exp,metric):
	
	'''
	Parameters:
	----------
	X : array - dataset
	t0 : inital time
	lifetime : final time of the process
	exp : power to which the metric is raised in order to obtain the 
		probability of extraction
	metric: string identifying the chosen metric 
		('variance','centroid_ratio','centroid_diff','min','min_corr')
		
	Returns:
	-------
	part,m : partitioning phase output
	list_p,list_m_leaf : merging phase output
	'''

	cut_ensemble = Matrix.cut_ensemble(X)
	
	print('PARTITIONING:')	 
	part,m = Partitioning.partitioning(cut_ensemble,t0,lifetime,metric,exp)
	
	print('MERGING:')
	list_p,list_m_leaf = Merging.merging(part,m,metric)
	list_p.reverse()
	list_m_leaf.reverse()
	
	return part,m,list_p,list_m_leaf




def save_tree(namefile,part,m,list_p,list_m_leaf):
	
	'''
	Save the mondrian_tree output in four .json files
	'''
	
	part.to_json(namefile+'_part.json')
	
	with open(namefile+'_m.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in m]))
	
	list_p_copy = list_p.copy()		
	for k in range(len(list_p_copy)):
		list_p_copy[k]['part_number'] = list_p_copy[k]['part_number'].astype(float)
		list_p_copy[k]['neighbors'] = [list(map(float, i)) for i in list_p_copy[k]['neighbors']]
		list_p_copy[k]['merged_part'] = [list(map(float, i)) for i in list_p_copy[k]['merged_part']]
	with open(namefile+'_p.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in list_p_copy]))
	
	list_m_leaf_copy = list_m_leaf.copy()
	for k in range(len(list_m_leaf_copy)):
		list_m_leaf_copy[k] = [i.to_dict() for i in list_m_leaf_copy[k]]
	with open(namefile+'_m_leaf.json', 'w') as f:
		f.write(json.dumps([df for df in list_m_leaf_copy]))

	return





def read_tree(namefile):
	
	'''
	Read the four .json files storing the mondrian_tree output
	'''
	
	part = json.load(open(namefile+'_part.json','r'))
	part = pd.DataFrame(part)
	part['polytope'] = [pc.Polytope(np.array(part['polytope'].iloc[i]['A']), np.array(part['polytope'].iloc[i]['b'])) for i in np.arange(len(part))]
	
	m = json.load(open(namefile+'_m.json','r'))
	m = [pd.DataFrame(i) for i in m]
	
	list_p = json.load(open(namefile+'_p.json','r'))
	list_p = [pd.DataFrame(i) for i in list_p]
	for k in range(len(list_p)):
		list_p[k]['part_number'] = list_p[k]['part_number'].astype(int)
		list_p[k]['neighbors'] = [list(map(int, i)) for i in list_p[k]['neighbors']]
		list_p[k]['merged_part'] = [list(map(int, i)) for i in list_p[k]['merged_part']]	
	
	list_m_leaf = json.load(open(namefile+'_m_leaf.json','r'))
	for k in range(len(list_m_leaf)):
		list_m_leaf[k] = [pd.DataFrame(i) for i in list_m_leaf[k]]
		list_m_leaf[k] = [i[i.columns[0:-1]].astype(int) for i in list_m_leaf[k]]
		
	return part,m,list_p,list_m_leaf




def class_assignment(list_m_leaf):

	classified_data = []
	for i in range(len(list_m_leaf)):
		classes = np.zeros(len(list_m_leaf[i][0]),dtype=int)
		indices = np.array(list_m_leaf[i][0]['index'])
		for j in range(1,len(list_m_leaf[i])):
			classes_j = j*np.ones(len(list_m_leaf[i][j]),dtype=int)
			classes = np.hstack([classes,classes_j])
			indices_j = np.array(list_m_leaf[i][j]['index'])
			indices = np.hstack([indices,indices_j])
		df = {'index':indices,'class':classes}
		df = pd.DataFrame(df)
		df = df.sort_values(by='index')
		classes_new = list(df['class'])
		classified_data.append(classes_new)	
		
	return classified_data




def ami(list_classified_data):
	
	number_of_trees = len(list_classified_data)
	pair = list(combinations(np.arange(number_of_trees),2))
	
	ami = []
	for k in range(len(pair)):
		ami_k = []
		index1 = pair[k][0]
		index2 = pair[k][1]
		tree1 = list_classified_data[index1].copy()
		tree2 = list_classified_data[index2].copy()
		for i in range(min(len(tree1),len(tree2))):
			labels1 = tree1[i]
			labels2 = tree2[i]
			ami_k.append(adjusted_mutual_info_score(labels1,labels2))
		ami.append(ami_k)
	
	ami_mean = list(pd.DataFrame(ami).mean())
	ami_std = list(pd.DataFrame(ami).std())	
	
	return ami_mean,ami_std,ami
	


	


def mondrian_forest(X,t0,lifetime,exp,metric,number_of_iterations):

	cut_ensemble = Matrix.cut_ensemble(X)
	
	list_part = []
	list_m = []
	list_p_tot = []
	list_m_leaf_tot = []
	class_data_tot = []
	for k in range(number_of_iterations):
		
		print('Tree number '+str(k+1))
		print('PARTITIONING:')	 
		part,m = Partitioning.partitioning(cut_ensemble,t0,lifetime,metric,exp)
		print('MERGING:')
		list_p,list_m_leaf = Merging.merging(part,m,metric)
		list_p.reverse()
		list_m_leaf.reverse()

		list_part.append(part)
		list_m.append(m)
		list_p_tot.append(list_p)
		list_m_leaf_tot.append(list_m_leaf)	

		#associo una classe
		classified_data = class_assignment(list_m_leaf)
		class_data_tot.append(classified_data)	
		
	ami_mean,ami_std,ami_tot = ami(class_data_tot)

		
	return list_part,list_m,list_p_tot,list_m_leaf_tot,ami_mean,ami_std,ami_tot





def save_forest(namefile,list_part,list_m,list_p_tot,list_m_leaf_tot,ami_mean,ami_std,ami_tot):
	
	'''
	Save the mondrian_forest output in .json files
	'''
	
	l = len(list_part)
	for i in range(l):
		part = list_part[i]
		m = list_m[i]
		list_p = list_p_tot[i]
		list_m_leaf = list_m_leaf_tot[i]
		save_tree(namefile+'_'+str(i),part,m,list_p,list_m_leaf)
	
	#save AMI
	df = {'AMI_mean':ami_mean,'AMI_std':ami_std}
	df = pd.DataFrame(df)
	df.to_csv(namefile+'_AMI.txt',sep='\t',index=False)
	
	with open(namefile+'_AMI.json', 'w') as f:
		f.write(json.dumps([k for k in ami_tot]))	

	return





def read_forest(namefile,number_of_iterations):

	'''
	Read the .json files storing the mondrian_forest output
	'''
	
	list_part = []
	list_m = []
	list_p_tot = []
	list_m_leaf_tot = []
	for i in range(number_of_iterations):
		part,m,list_p,list_m_leaf = read_tree(namefile+'_'+str(i))
		list_part.append(part)
		list_m.append(m)
		list_p_tot.append(list_p)
		list_m_leaf_tot.append(list_m_leaf)
	
	ami_tot = json.load(open(namefile+'_AMI.json','r'))
	df = pd.read_csv(namefile+'_AMI.txt',sep='\t')
	ami_mean = list(df['AMI_mean'])
	ami_std = list(df['AMI_std'])
	
	return list_part,list_m,list_p_tot,list_m_leaf_tot,ami_mean,ami_std,ami_tot

#namefile = name+'_lambda'+str(lifetime)+'_exp'+str(exp)+'_'+metric+'_'+str(k+1)
	