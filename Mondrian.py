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
	data = cut_ensemble[0].copy()
	
	print('PARTITIONING:')	 
	part,m = Partitioning.partitioning(cut_ensemble,t0,lifetime,metric,exp)
	
	print('MERGING:')
	list_p,classified_data = Merging.merging(part,m,metric,data)
	list_p.reverse()
	classified_data.reverse()
	
	return data,part,m,list_p,classified_data




def save_tree(namefile,data,part,m,list_p,classified_data):
	
	'''
	Save the mondrian_tree output in four .json files and one .txt file
	'''
	
	data.to_csv(namefile+'_data.txt',sep='\t',index=False)
	
	part.to_json(namefile+'_part.json')
	
	with open(namefile+'_m.json', 'w') as f:
		f.write(json.dumps(m))
	
	list_p_copy = list_p.copy()		
	for k in range(len(list_p_copy)):
		list_p_copy[k]['part_number'] = list_p_copy[k]['part_number'].astype(float)
		list_p_copy[k]['neighbors'] = [list(map(float, i)) for i in list_p_copy[k]['neighbors']]
		list_p_copy[k]['merged_part'] = [list(map(float, i)) for i in list_p_copy[k]['merged_part']]
	with open(namefile+'_p.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in list_p_copy]))
	
	with open(namefile+'_classified_data.json', 'w') as f:
		f.write(json.dumps(classified_data))
		
	return





def read_tree(namefile):
	
	'''
	Read the four .json files storing the mondrian_tree output
	'''
	
	data = pd.read_csv(namefile+'_data.txt',sep='\t')
	
	part = json.load(open(namefile+'_part.json','r'))
	part = pd.DataFrame(part)
	part['polytope'] = [pc.Polytope(np.array(part['polytope'].iloc[i]['A']), np.array(part['polytope'].iloc[i]['b'])) for i in np.arange(len(part))]
	
	m = json.load(open(namefile+'_m.json','r'))
	
	list_p = json.load(open(namefile+'_p.json','r'))
	list_p = [pd.DataFrame(i) for i in list_p]
	for k in range(len(list_p)):
		list_p[k]['part_number'] = list_p[k]['part_number'].astype(int)
		list_p[k]['neighbors'] = [list(map(int, i)) for i in list_p[k]['neighbors']]
		list_p[k]['merged_part'] = [list(map(int, i)) for i in list_p[k]['merged_part']]	
	
	classified_data = json.load(open(namefile+'_classified_data.json','r'))
	
	return data,part,m,list_p,classified_data







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
	data = cut_ensemble[0].copy()
	
	list_part = []
	list_m = []
	list_p_tot = []
	list_classified_data = []
	for k in range(number_of_iterations):
		
		print('Tree number '+str(k+1))
		print('PARTITIONING:')	 
		part,m = Partitioning.partitioning(cut_ensemble,t0,lifetime,metric,exp)
		print('MERGING:')
		list_p,classified_data = Merging.merging(part,m,metric,data)
		list_p.reverse()
		classified_data.reverse()

		list_part.append(part)
		list_m.append(m)
		list_p_tot.append(list_p)
		list_classified_data.append(classified_data)	
		
	ami_mean,ami_std,ami_tot = ami(list_classified_data)

		
	return data,list_part,list_m,list_p_tot,list_classified_data,ami_mean,ami_std,ami_tot#,list_m_leaf_tot





def save_forest(namefile,data,list_part,list_m,list_p_tot,list_classified_data,ami_mean,ami_std,ami_tot):
	
	'''
	Save the mondrian_forest output in .json files
	'''
	
	data.to_csv(namefile+'_data.txt',sep='\t',index=False)
	
	l = len(list_part)
	for i in range(l):
		part = list_part[i]
		m = list_m[i]
		list_p = list_p_tot[i]
		classified_data = list_classified_data[i]
		save_tree(namefile+'_'+str(i),data,part,m,list_p,classified_data)
	
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
	
	data = pd.read_csv(namefile+'_data.txt',sep='\t')
	
	list_part = []
	list_m = []
	list_p_tot = []
	list_m_leaf_tot = []
	for i in range(number_of_iterations):
		
		part = json.load(open(namefile+'_'+str(i)+'_part.json','r'))
		part = pd.DataFrame(part)
		part['polytope'] = [pc.Polytope(np.array(part['polytope'].iloc[i]['A']), np.array(part['polytope'].iloc[i]['b'])) for i in np.arange(len(part))]
	
		m = json.load(open(namefile+'_'+str(i)+'_m.json','r'))
	
		list_p = json.load(open(namefile+'_'+str(i)+'_p.json','r'))
		list_p = [pd.DataFrame(i) for i in list_p]
		for k in range(len(list_p)):
			list_p[k]['part_number'] = list_p[k]['part_number'].astype(int)
			list_p[k]['neighbors'] = [list(map(int, i)) for i in list_p[k]['neighbors']]
			list_p[k]['merged_part'] = [list(map(int, i)) for i in list_p[k]['merged_part']]	
	
		classified_data = json.load(open(namefile+'_'+str(i)+'_classified_data.json','r'))

		list_part.append(part)
		list_m.append(m)
		list_p_tot.append(list_p)
		list_classified_data.append(classified_data)
	
	ami_tot = json.load(open(namefile+'_AMI.json','r'))
	df = pd.read_csv(namefile+'_AMI.txt',sep='\t')
	ami_mean = list(df['AMI_mean'])
	ami_std = list(df['AMI_std'])
	
	return data,list_part,list_m,list_p_tot,list_classified_data,ami_mean,ami_std,ami_tot


#namefile = name+'_lambda'+str(lifetime)+'_exp'+str(exp)+'_'+metric+'_'+str(k+1)
	
'''
def class_assignment(list_m_leaf):

	classified_data = []
	for i in range(len(list_m_leaf)):
		classes = np.zeros(len(list_m_leaf[i][0]),dtype=int)
		indices = np.array(list_m_leaf[i][0]).copy()#['index'])
		for j in range(1,len(list_m_leaf[i])):
			classes_j = j*np.ones(len(list_m_leaf[i][j]),dtype=int)
			classes = np.hstack([classes,classes_j])
			indices_j = np.array(list_m_leaf[i][j]).copy()#['index'])
			indices = np.hstack([indices,indices_j])
		df = {'index':indices,'class':classes}
		df = pd.DataFrame(df)
		df = df.sort_values(by='index')
		classes_new = list(df['class'])
		classified_data.append(classes_new)	
		
	return classified_data
'''

