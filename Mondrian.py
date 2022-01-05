import pandas as pd
import numpy as np
import json
from sklearn.metrics.cluster import adjusted_mutual_info_score
from itertools import combinations
import copy

import Matrix
import Partitioning
import Merging



def mondrian_tree(namefile,X,t0,lifetime,exp,metric):

	dist_matrix = Matrix.distance_matrix(X)

	print('PARTITIONING:')	 
	m,part = Partitioning.partitioning(X,t0,lifetime,dist_matrix,metric,exp)

	#save file
	#part
	part.to_json(namefile+'_part.json')
	#m
	lista = list(np.array(m,dtype=object)[:,3])
	for i in lista:
		i.columns = i.columns.astype(str)
	with open(namefile+'_m.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in lista]))

	part = json.load(open(namefile+'_part.json','r'))
	part = pd.DataFrame(part)
	m = json.load(open(namefile+'_m.json','r'))
	
	print('MERGING:')
	list_m_leaf,list_p = Merging.merging(m,part,metric)
	list_p.reverse()
	list_m_leaf.reverse()
	
	#save file
	#p		
	with open(namefile+'_p.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in list_p]))
	#m_leaf
	with open(namefile+'_m_leaf.json', 'w') as f:
		f.write(json.dumps([df for df in list_m_leaf]))
		
	return 




def read_tree(namefile):
	
	part = json.load(open(namefile+'_part.json','r'))
	part = pd.DataFrame(part)
	m = json.load(open(namefile+'_m.json','r'))
	list_p = json.load(open(namefile+'_p.json','r'))
	list_m_leaf = json.load(open(namefile+'_m_leaf.json','r'))

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
	
	coeff_medio = pd.DataFrame(coeff_tot).mean()
	coeff_std = pd.DataFrame(coeff_tot).std()	
	
	return coeff_medio,coeff_std,coeff_tot
	




def mondrian_forest(name,X,t0,lifetime,exp,metric,number_of_iterations):
	
	list_m_leaf_tot = []
	class_data_tot = []
	for k in range(number_of_iterations):
		
		print('Tree number '+str(k+1))
		namefile = name+'_lambda'+str(lifetime)+'_exp'+str(exp)+'_'+metric+'_'+str(k+1)
		mondrian_tree(namefile,X,t0,lifetime,exp,metric)
		
		#leggo file	
		part,m,list_p,list_m_leaf = read_tree(namefile)
		list_m_leaf_tot.append(list_m_leaf)	

		#associo una classe
		classified_data = class_assignment(list_m_leaf)
		class_data_tot.append(classified_data)	
		
	c_mean,c_std,c_tot = ami(class_data_tot)

	#save AMI
	df = {'AMI_mean':c_mean,'AMI_std':c_std}
	df = pd.DataFrame(df)
	df.to_csv(name+'_AMI.txt',sep='\t',index=False)
	
	with open(name+'_AMI.json', 'w') as f:
		f.write(json.dumps([l for l in c_tot]))	

		
	return




	