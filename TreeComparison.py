# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from itertools import combinations
import copy
import matplotlib.pylab as plt 



def AssignClass(list_m_leaf):

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





def AMI(class_data_tot_true):#,name_file
	
	class_data_tot = copy.deepcopy(class_data_tot_true)
	
	pair = list(combinations(np.arange(len(class_data_tot)),2))
	
	coeff_tot = []
	for k in range(len(pair)):
	
		coeff=[]
		index1 = pair[k][0]
		index2 = pair[k][1]
		for i in range(min(len(class_data_tot[index1]),len(class_data_tot[index2]))):
			cl1 = class_data_tot[index1][i]
			cl1.columns = ['index','class1']
			cl2 = class_data_tot[index2][i]
			cl2.columns = ['index','class2']
			df = pd.merge(cl1,cl2,left_on='index',right_on='index',how='inner')
			#coeff.append(adjusted_mutual_info_score(cl1['class'],cl2['class']))
			coeff.append(adjusted_mutual_info_score(df['class1'],df['class2']))
			
		coeff_tot.append(coeff)
	
	coeff_medio = pd.DataFrame(coeff_tot).mean()
	coeff_std = pd.DataFrame(coeff_tot).std()

	fig,ax = plt.subplots()
	ax.plot(np.arange(2,len(coeff_medio)+1),coeff_medio[1:],linewidth=0.7)	
	ax.scatter(np.arange(2,len(coeff_medio)+1),coeff_medio[1:],s=10)
	ax.fill_between(np.arange(2,len(coeff_medio)+1), coeff_medio[1:]-coeff_std[1:]/2, coeff_medio[1:]+coeff_std[1:]/2,alpha=0.2,color='b')
	ax.set_xlabel('Number of Clusters')
	ax.set_ylabel('Adjusted Mutual Information')

	#if name_file != False:
	#	plt.savefig(name_file)	
	
	return coeff_medio,coeff_std,coeff_tot
	






def FMI(class_data_tot,y,number_of_clusters):
	
	y = pd.DataFrame(y)
	y['index'] = y.index
	y.columns = ['class_y','index']
	
	coeff = []
	for i in range(len(class_data_tot)):
		df = class_data_tot[i][number_of_clusters-1]
		df = pd.merge(df,y,left_on='index',right_on='index',how='inner')
		#coeff.append(accuracy_score(df['class_y'],df['class']))
		coeff.append(fowlkes_mallows_score(df['class_y'],df['class']))

	return coeff


