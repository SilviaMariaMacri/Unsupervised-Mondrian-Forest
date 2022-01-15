import pandas as pd
import numpy as np
import json
from sklearn.metrics.cluster import adjusted_mutual_info_score
from itertools import combinations
import copy
import polytope as pc

import Matrix
import Partitioning
import Merging



def mondrian_tree(X,t0,lifetime,exp,metric):

	dist_matrix = Matrix.distance_matrix(X)

	print('PARTITIONING:')	 
	m,part = Partitioning.partitioning(X,t0,lifetime,dist_matrix,metric,exp)
	
	print('MERGING:')
	list_m_leaf,list_p = Merging.merging(m,part,metric)
	list_p.reverse()
	list_m_leaf.reverse()
	
	return part,m,list_p,list_m_leaf




def save_tree(namefile,part,m,list_p,list_m_leaf):
	
	part.to_json(namefile+'_part.json')
	
	with open(namefile+'_m.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in m]))
			
	with open(namefile+'_p.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in list_p]))
	
	for k in range(len(list_m_leaf)):
		list_m_leaf[k] = [i.to_dict() for i in list_m_leaf[k]]
	with open(namefile+'_m_leaf.json', 'w') as f:
		f.write(json.dumps([df for df in list_m_leaf]))

	return



def read_tree(namefile):
	
	part = json.load(open(namefile+'_part.json','r'))
	part = pd.DataFrame(part)
	part['polytope'] = [pc.Polytope(np.array(part['polytope'].iloc[i]['A']), np.array(part['polytope'].iloc[i]['b'])) for i in np.arange(len(part))]
	
	m = json.load(open(namefile+'_m.json','r'))
	m = [pd.DataFrame(i) for i in m]
	
	list_p = json.load(open(namefile+'_p.json','r'))
	list_p = [pd.DataFrame(i) for i in list_p]
	
	list_m_leaf = json.load(open(namefile+'_m_leaf.json','r'))
	for k in range(len(list_m_leaf)):
		list_m_leaf[k] = [pd.DataFrame(i) for i in list_m_leaf[k]]
		
	return part,m,list_p,list_m_leaf




def class_assignment(list_m_leaf):

	classified_data = []
	for i in range(len(list_m_leaf)):
		classe = np.arange(len(list_m_leaf[i]))
		df = pd.DataFrame()
		for j in range(len(list_m_leaf[i])):
			df_j = pd.DataFrame(list_m_leaf[i][j])
			df_j['class'] = classe[j]
			df = pd.concat([df,df_j])
		df = df[['index','class']]
		df.index = np.arange(len(df))
		classified_data.append(df)
		
	return classified_data





def ami(list_classified_data_input):
	
	list_classified_data = copy.deepcopy(list_classified_data_input)
	
	pair = list(combinations(np.arange(len(list_classified_data)),2))
	
	coeff_tot = []
	for k in range(len(pair)):
	
		coeff=[]
		index1 = pair[k][0]
		index2 = pair[k][1]
		for i in range(min(len(list_classified_data[index1]),len(list_classified_data[index2]))):
			cl1 = list_classified_data[index1][i]
			cl1.columns = ['index','class1']
			cl2 = list_classified_data[index2][i]
			cl2.columns = ['index','class2']
			df = pd.merge(cl1,cl2,left_on='index',right_on='index',how='inner')
			#coeff.append(adjusted_mutual_info_score(cl1['class'],cl2['class']))
			coeff.append(adjusted_mutual_info_score(df['class1'],df['class2']))
			
		coeff_tot.append(coeff)
	
	coeff_medio = list(pd.DataFrame(coeff_tot).mean())
	coeff_std = list(pd.DataFrame(coeff_tot).std())	
	
	return coeff_medio,coeff_std,coeff_tot
	




def mondrian_forest(X,t0,lifetime,exp,metric,number_of_iterations):
	
	list_part = []
	list_m = []
	list_p_tot = []
	list_m_leaf_tot = []
	class_data_tot = []
	for k in range(number_of_iterations):
		
		print('Tree number '+str(k+1))
		part,m,list_p,list_m_leaf = mondrian_tree(X,t0,lifetime,exp,metric)
		
		list_part.append(part)
		list_m.append(m)
		list_p_tot.append(list_p)
		list_m_leaf_tot.append(list_m_leaf)	

		#associo una classe
		classified_data = class_assignment(list_m_leaf)
		class_data_tot.append(classified_data)	
		
	c_mean,c_std,c_tot = ami(class_data_tot)

		
	return list_part,list_m,list_p_tot,list_m_leaf_tot,c_mean,c_std,c_tot



def save_forest(namefile,list_part,list_m,list_p_tot,list_m_leaf_tot,c_mean,c_std,c_tot):
	
	l = len(list_part)
	for i in range(l):
		part = list_part[i]
		m = list_m[i]
		list_p = list_p_tot[i]
		list_m_leaf = list_m_leaf_tot[i]
		save_tree(namefile+'_'+str(i),part,m,list_p,list_m_leaf)
	
	#save AMI
	df = {'AMI_mean':c_mean,'AMI_std':c_std}
	df = pd.DataFrame(df)
	df.to_csv(namefile+'_AMI.txt',sep='\t',index=False)
	
	with open(namefile+'_AMI.json', 'w') as f:
		f.write(json.dumps([k for k in c_tot]))	

	return


def read_forest(namefile,number_of_iterations):

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
	
	c_tot = json.load(open(namefile+'_AMI.json','r'))
	df = pd.read_csv(namefile+'_AMI.txt',sep='\t')
	c_mean = list(df['AMI_mean'])
	c_std = list(df['AMI_std'])
	
	return list_part,list_m,list_p_tot,list_m_leaf_tot,c_mean,c_std,c_tot

#namefile = name+'_lambda'+str(lifetime)+'_exp'+str(exp)+'_'+metric+'_'+str(k+1)
	