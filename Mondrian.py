import pandas as pd
import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score
from itertools import combinations
import polytope as pc
import json
import copy

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
	part_space,part_data = Partitioning.partitioning(cut_ensemble,t0,lifetime,metric,exp)
	
	print('MERGING:')
	merg_space,merg_data = Merging.merging(part_space,part_data,metric,data)
	merg_space.reverse()
	merg_data.reverse()
	
	return data,part_space,part_data,merg_space,merg_data




def save_tree(namefile,data,part_space,part_data,merg_space,merg_data):
	
	'''
	Save the mondrian_tree output in four .json files and one .txt file
	'''
	
	with open(namefile+'_data.json', 'w') as f:
		f.write(json.dumps(data.tolist()))
	
	part_space.to_json(namefile+'_part_space.json')
	
	part_data = [i.tolist() for i in part_data]
	with open(namefile+'_part_data.json', 'w') as f:
		f.write(json.dumps(part_data))
	
	merg_space_copy = copy.deepcopy(merg_space)		
	for k in range(len(merg_space_copy)):
		merg_space_copy[k]['id_number'] = merg_space_copy[k]['id_number'].astype(float)
		merg_space_copy[k]['neighbors'] = [list(map(float, i)) for i in merg_space_copy[k]['neighbors']]
		merg_space_copy[k]['merged'] = [list(map(float, i)) for i in merg_space_copy[k]['merged']]
	with open(namefile+'_merg_space.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in merg_space_copy]))
	
	with open(namefile+'_merg_data.json', 'w') as f:
		f.write(json.dumps(merg_data))
		
	return





def read_tree(namefile):
	
	'''
	Read the four .json files storing the mondrian_tree output
	'''
	
	data = json.load(open(namefile+'_data.json','r'))
	data = np.array(data)
	
	part_space = json.load(open(namefile+'_part_space.json','r'))
	part_space = pd.DataFrame(part_space)
	part_space['polytope'] = [pc.Polytope(np.array(part_space['polytope'].iloc[i]['A']), np.array(part_space['polytope'].iloc[i]['b'])) for i in np.arange(len(part_space))]
	
	part_data = json.load(open(namefile+'_part_data.json','r'))
	part_data = [np.array(i) for i in part_data]
	
	merg_space = json.load(open(namefile+'_merg_space.json','r'))
	merg_space = [pd.DataFrame(i) for i in merg_space]
	for k in range(len(merg_space)):
		merg_space[k]['id_number'] = merg_space[k]['id_number'].astype(np.int64)
		merg_space[k]['neighbors'] = [list(map(np.int64, i)) for i in merg_space[k]['neighbors']]
		merg_space[k]['merged'] = [list(map(int, i)) for i in merg_space[k]['merged']]	
	
	merg_data = json.load(open(namefile+'_merg_data.json','r'))
	
	return data,part_space,part_data,merg_space,merg_data







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
	
	part_space_list = []
	part_data_list = []
	merg_space_list = []
	merg_data_list = []
	for k in range(number_of_iterations):
		
		print('Tree number '+str(k+1))
		print('PARTITIONING:')	 
		part_space,part_data = Partitioning.partitioning(cut_ensemble,t0,lifetime,metric,exp)
		print('MERGING:')
		merg_space,merg_data = Merging.merging(part_space,part_data,metric,data)
		merg_space.reverse()
		merg_data.reverse()

		part_space_list.append(part_space)
		part_data_list.append(part_data)
		merg_space_list.append(merg_space)
		merg_data_list.append(merg_data)	
		
	ami_mean,ami_std,ami_tot = ami(merg_data_list)

		
	return data,part_space_list,part_data_list,merg_space_list,merg_data_list,ami_mean,ami_std,ami_tot





def save_forest(namefile,starting_namefile_number,data,part_space_list,part_data_list,merg_space_list,merg_data_list,ami_mean,ami_std,ami_tot):
	
	'''
	Save the mondrian_forest output in .json files
	'''
	
	with open(namefile+'_data.json', 'w') as f:
		f.write(json.dumps(data.tolist()))
			
	l = len(part_space_list)
	for i in range(l):
		
		name = namefile+'_'+str(starting_namefile_number+i)
		part_space = part_space_list[i]
		part_data = part_data_list[i]
		merg_space = merg_space_list[i]
		merg_data = merg_data_list[i]

		part_space.to_json(name+'_part_space.json')
		
		part_data = [i.tolist() for i in part_data]
		with open(name+'_part_data.json', 'w') as f:
			f.write(json.dumps(part_data))
		
		merg_space_copy = copy.deepcopy(merg_space)		
		for k in range(len(merg_space_copy)):
			merg_space_copy[k]['id_number'] = merg_space_copy[k]['id_number'].astype(float)
			merg_space_copy[k]['neighbors'] = [list(map(float, i)) for i in merg_space_copy[k]['neighbors']]
			merg_space_copy[k]['merged'] = [list(map(float, i)) for i in merg_space_copy[k]['merged']]
		with open(name+'_merg_space.json', 'w') as f:
			f.write(json.dumps([df.to_dict() for df in merg_space_copy]))
		
		with open(name+'_merg_data.json', 'w') as f:
			f.write(json.dumps(merg_data))

	#save AMI
	df = {'AMI_mean':ami_mean,'AMI_std':ami_std}
	df = pd.DataFrame(df)
	df.to_csv(namefile+'_AMI.txt',sep='\t',index=False)
	
	with open(namefile+'_AMI.json', 'w') as f:
		f.write(json.dumps([k for k in ami_tot]))	

	return





def read_forest(namefile,starting_namefile_number,number_of_iterations):

	'''
	Read the .json files storing the mondrian_forest output
	'''
	
	data = json.load(open(namefile+'_data.json','r'))
	data = np.array(data)
	
	part_space_list = []
	part_data_list = []
	merg_space_list = []
	merg_data_list = []
	for i in range(number_of_iterations):
		name = namefile+'_'+str(starting_namefile_number+i)
				
		part_space = json.load(open(name+'_part_space.json','r'))
		part_space = pd.DataFrame(part_space)
		part_space['polytope'] = [pc.Polytope(np.array(part_space['polytope'].iloc[i]['A']), np.array(part_space['polytope'].iloc[i]['b'])) for i in np.arange(len(part_space))]
		
		part_data = json.load(open(name+'_part_data.json','r'))
		part_data = [np.array(i) for i in part_data]
		
		merg_space = json.load(open(name+'_merg_space.json','r'))
		merg_space = [pd.DataFrame(i) for i in merg_space]
		for k in range(len(merg_space)):
			merg_space[k]['id_number'] = merg_space[k]['id_number'].astype(np.int64)
			merg_space[k]['neighbors'] = [list(map(np.int64, i)) for i in merg_space[k]['neighbors']]
			merg_space[k]['merged'] = [list(map(int, i)) for i in merg_space[k]['merged']]	
		
		merg_data = json.load(open(name+'_merg_data.json','r'))
		
		part_space_list.append(part_space)
		part_data_list.append(part_data)
		merg_space_list.append(merg_space)
		merg_data_list.append(merg_data)
	
	ami_tot = json.load(open(namefile+'_AMI.json','r'))
	df = pd.read_csv(namefile+'_AMI.txt',sep='\t')
	ami_mean = list(df['AMI_mean'])
	ami_std = list(df['AMI_std'])
	
	return data,part_space_list,part_data_list,merg_space_list,merg_data_list,ami_mean,ami_std,ami_tot


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

