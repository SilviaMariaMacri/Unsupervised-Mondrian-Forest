
data_classification = pd.DataFrame(list_class[0][number_of_clusters-1])
data_classification.columns = ['index','class0']
number_of_clusters = 4
for i in range(1,len(list_class)):
	data = pd.DataFrame(list_class[i][number_of_clusters-1])
	data.columns = ['index','class'+str(i)]
	data_classification = pd.merge(data_classification,data,right_on='index',left_on='index')
	




#%%

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist,pdist
import networkx as nx
from sklearn.metrics.cluster import adjusted_mutual_info_score
from itertools import combinations




#Mondrian tagli paralleli
def trova_part_vicine(part):


	neighbors = []
	
	for i in range(len(part.query('leaf==True'))):
		
		p = part.query('leaf==True').copy()
		p.index = np.arange(len(part.query('leaf==True')))
		
		for j in range(2):
			p['min'+str(j)+'_'+str(i)] = p['min'+str(j)].iloc[i]
			p['max'+str(j)+'_'+str(i)] = p['max'+str(j)].iloc[i]
			
		for j in range(2):	
			p=(p.eval('vicinoA'+str(j)+'_'+str(i)+' = ((min'+str(j)+'>=min'+str(j)+'_'+str(i)+') and (min'+str(j)+'<=max'+str(j)+'_'+str(i)+')) or ( (max'+str(j)+'>=min'+str(j)+'_'+str(i)+') and (max'+str(j)+'<=max'+str(j)+'_'+str(i)+'))')
		 .eval('vicinoB'+str(j)+'_'+str(i)+' = ((min'+str(j)+'_'+str(i)+'>=min'+str(j)+') and (min'+str(j)+'_'+str(i)+'<=max'+str(j)+')) or ( (max'+str(j)+'_'+str(i)+'>=min'+str(j)+') and (max'+str(j)+'_'+str(i)+'<=max'+str(j)+'))'))
			
			#p=p.query('vicino'+str(j)+'_'+str(i)+'==True')
			p=p.query('(vicinoA'+str(j)+'_'+str(i)+'==True) or (vicinoB'+str(j)+'_'+str(i)+'==True)')
			
			
		p = p.drop(i)
		
		neighbors.append(list(p['part_number']))
	
	
	
	df={'part_number':part.query('leaf==True')['part_number'],'neighbors':neighbors}
	df=pd.DataFrame(df)
	df.index = np.arange(len(df))
	



	return df
# per più di due dimensioni?
# come generalizzarla a partizione con tagli non regolari?



def calcolo_varianza_part_vicine(data1,data2):
	

	data1 = data1.drop('index',axis=1)
	data2 = data2.drop('index',axis=1)
	
	pd1 = pdist(data1)
	pd2 = pdist(data2)
	
	data = np.vstack([np.array(data1),np.array(data2)])
	pd = pdist(data)
	pd12 = np.hstack([pd1, pd2])
	
	var_part_unica = np.var(pd)
	#var1 = np.var(pd1)
	#var2 = np.var(pd2)
	var_entrambe_separate = np.var(pd12) 


	return var_part_unica,var_entrambe_separate#,var1,var2




def Centroid_part_vicine(data1,data2):

	
	data1 = np.array(data1.drop('index',axis=1))
	data2 = np.array(data2.drop('index',axis=1))
	
	data12 = np.vstack([data1,data2])
	
		
	data_tot = [data1,data2,data12]
	centr=[[],[],[]]
	for j in range(3):
		for i in range(len(data12[0])):
			centr[j].append(np.mean(data_tot[j][:,i]))
	
	dist=[]
	for i in range(3):
		dist.append(cdist(data_tot[i],[centr[i]]))
	
	#mean_dist12 = np.mean(dist[2])
	#mean_dist1 = np.mean(dist[0])
	#mean_dist2 = np.mean(dist[1])	
	ratio = np.mean(dist[2])/np.mean(np.vstack([dist[0],dist[1]]))	
	difference = np.mean(dist[2]) - np.mean(np.vstack([dist[0],dist[1]]))
	
	#data1_number = len(data1)
	#data2_number = len(data2)


	return ratio,difference#,mean_dist1,mean_dist2,mean_dist12,data1_number,data2_number	  



#in MergePart data1 corrisponde alla partizione con dato singolo


def MinDistBetweenPart(data1,data2):#,tagli_paralleli
	
	data1 = data1.drop('index',axis=1) 
	data2 = data2.drop('index',axis=1)
	
	#if tagli_paralleli == True:
	#	data1 = data1.drop('part_number',axis=1) 
	#	data2 = data2.drop('part_number',axis=1)
		
	data1 = np.array(data1)
	data2 = np.array(data2)
	
	pd1 = cdist(data1,data1)
	pd2 = cdist(data2,data2)
			
	min1 = np.min(np.where(pd1!= 0, pd1, np.inf),axis=0)
	min2 = np.min(np.where(pd2!= 0, pd2, np.inf),axis=0)
			
	min_tot = np.hstack([min1,min2])
	if np.inf in min_tot:
		min_tot = list(min_tot)
		min_tot.remove(np.inf)
	media = np.mean(min_tot)
	mean2 = np.mean(min2)
		
	dist = cdist(data1,data2)
	#data1 = righe
	#data2 = colonne
	min_dist_fra_partizioni = dist.min()
	ind = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
	
	dist_point2 = pd2[ind[1]]
	dist_point2 = np.delete(dist_point2,ind[1])
	min_data2 = np.min(dist_point2)
		
	try:
		dist_point1 = pd1[ind[0]]
		dist_point1 = np.delete(dist_point1,ind[0])
		min_data1 = np.min(dist_point1)
		mean1 = np.mean(min1)	
	except ValueError:
		return media,min_dist_fra_partizioni,min_data2,mean2
	
	return media,min_dist_fra_partizioni,min_data1,min_data2,mean1,mean2	  



# assegnare classi a punti
def AssignClass(G,X,p,m_leaf,tagli_paralleli):
	
	
	if tagli_paralleli==True:
		a = AssignPartition(X,part)
	data = pd.DataFrame()
	for i in range(len(p)):
		if tagli_paralleli == True:
			data_part = a.query('part_number=='+str(p.iloc[i]['part_number'])).copy()
		else:
			#data_part = m[p.iloc[i]['part_number']][2]
			data_part = pd.DataFrame(m_leaf[i])
			data_part['part_number'] = int(p.iloc[i]['part_number'])*np.ones(len(data_part)).astype(int)
			#print(len(data_part))
		data = pd.concat([data,data_part])
	data.index = np.arange(len(data))
	
	list_conn_comp = list(nx.connected_components(G))
	data['class'] = 0
	cl = np.arange(len(list_conn_comp))
	for i in range(len(list_conn_comp)):
		for j in list_conn_comp[i]:
			#print(j)
			data.loc[data.query('part_number=='+str(j)).index, 'class'] = cl[i]
	
		
	return data



def MergePart(m_leaf_true,p_true,part_to_remove,part_to_merge):
	
	p = p_true.copy(deep=True)
	m_leaf = m_leaf_true.copy()
	index1 = p[p['part_number']==part_to_remove].index[0]
	index2 = p[p['part_number']==part_to_merge].index[0]
	#unisco i dati in m_leaf
	data1 = pd.DataFrame(m_leaf[index1])
	
	data2 = pd.DataFrame(m_leaf[index2])
	data2 = pd.concat([data2,data1])
	data2.index = np.arange(len(data2))
	m_leaf[index2] = data2.to_dict()
	
	#neigh = list(p['neighbors'])
	#merged_part = list(p['merged_part'])
			
	#sostituisco vicini in tutte le partizioni di p e aggiungo vicini di single_part a part unita
	p['neighbors'].iloc[index1].remove(part_to_merge)
	p['neighbors'].iloc[index2].remove(part_to_remove)
	for j in p['neighbors'].iloc[index1]:
		if j not in p['neighbors'].iloc[index2]:
			p['neighbors'].iloc[index2].append(j)
		p['neighbors'].iloc[p[p['part_number']==j].index[0]].remove(part_to_remove)
		if part_to_merge not in p['neighbors'].iloc[p[p['part_number']==j].index[0]]:
			p['neighbors'].iloc[p[p['part_number']==j].index[0]].append(part_to_merge)
	
	for j in p['merged_part'].iloc[index1]:
		p['merged_part'].iloc[index2].append(j)
	p['merged_part'].iloc[index2].append(part_to_remove)
	p = p.drop(index1)
	p.index = np.arange(len(p))
	m_leaf = np.delete(m_leaf, index1).tolist()
	
	return m_leaf,p
	
	





	

#unisce partizione con solo un dato a quella più vicina
def MergePart_SingleData(m,part): 
	
	p = part.copy(deep=True)
	p = p.query('leaf==True')
	p = p[['part_number','neighbors']]
	merged_part = []
	for i in range(len(p)):
		merged_part.append([])
	p['merged_part'] = merged_part
	p.index = np.arange(len(p))
	m_leaf = m.copy()
	m_leaf = np.delete(m, list(part.query('leaf==False')['part_number'])).tolist()
	
	#list_part_with_single_data = []
	#list_part_to_merge = []
	list_part = list(p['part_number'])
	for i in list_part:
		index1 = p[p['part_number']==i].index[0]
		data1 = pd.DataFrame(m_leaf[index1])
		#consdidero solo partizioni con unico dato all'interno
		if len(data1) == 1: 
			#cerco minima distanza con part vicine
			min_dist = []
			for j in p['neighbors'].iloc[index1]:
				data2 = pd.DataFrame(m_leaf[p[p['part_number']==j].index[0]])
				if len(data2) > 1:
					media,min_dist_fra_partizioni,min_data2,mean2 = MinDistBetweenPart(data1,data2)	
					min_dist.append(min_dist_fra_partizioni + min_data2)
				else:
					min_dist.append(np.inf)
			minimo_fra_minimi = min(min_dist)
			index_nearest_part = min_dist.index(minimo_fra_minimi)
			part_to_merge = p['neighbors'].iloc[index1][index_nearest_part]
			#list_part_with_single_data.append(i)
			#list_part_to_merge.append(part_to_merge)
			
			
			m_leaf,p = MergePart(m_leaf,p,i,part_to_merge)
			
			'''
			#unisco i dati in m_leaf
			data2 = pd.DataFrame(m_leaf[p[p['part_number']==part_to_merge].index[0]])
			data2 = pd.concat([data2,data1])
			data2.index = np.arange(len(data2))
			m_leaf[p[p['part_number']==part_to_merge].index[0]] = data2.to_dict()
			
			#sostituisco vicini in tutte le partizioni di p e aggiungo vicini di single_part a part unita
			p['neighbors'].iloc[i].remove(part_to_merge)
			p['neighbors'].iloc[p[p['part_number']==part_to_merge].index[0]].remove(part_with_single_data)
			for j in p['neighbors'].iloc[i]:
				if j not in p['neighbors'].iloc[p[p['part_number']==part_to_merge].index[0]]:
					p['neighbors'].iloc[p[p['part_number']==part_to_merge].index[0]].append(j)
				p['neighbors'].iloc[p[p['part_number']==j].index[0]].remove(part_with_single_data)
				if part_to_merge not in p['neighbors'].iloc[p[p['part_number']==j].index[0]]:
					p['neighbors'].iloc[p[p['part_number']==j].index[0]].append(part_to_merge)
			

	#neigh = list(p['neighbors'])
	#neigh_new = []
	#for i in range(len(neigh)):
	#	neigh_new.append(list(set(neigh[i])))
	#p = p.drop('neighbors',axis=1)
	#p['neighbors'] = neigh_new
	p = p.drop(merged_part['index'])
	p.index = np.arange(len(p))
	m_leaf = np.delete(m_leaf, list(merged_part['index'])).tolist()
	'''
	
	#merged_part = {'part_single_data':list_part_with_single_data,'part_to_merge':list_part_to_merge}#'index':index,
	#merged_part = pd.DataFrame(merged_part)
	
		
	return m_leaf,p #merged_part,




#print
# score = 'var','centroid','min'
#tagli_paralleli = True,False
def PartLinkScore(m_leaf,p,score):#X,,tagli_paralleli
	
	# è utile tener conto del numero di punti contenuti in una partizione? ()
	#merged_partitions,p,m_red = MergePart(m,part)
	part1 = []
	part2 = []
	#p = part.query('leaf==True').copy()
	#p.index = np.arange(len(p))
	# tagli paralleli
	'''
	if tagli_paralleli ==  True:
		df = trova_part_vicine(part)
		p = pd.merge(p,df,right_on='part_number',left_on='part_number')
		a = AssignPartition(X,part)
	'''	
	v_unica = []
	v_sep = []
	ratio_centroid = []
	difference_centroid = []
	diff_minimi = []
	'''
	list_min_dist_i = []
	list_min_data1=[]
	list_min_data2=[]
	list_mean1=[]
	list_mean2=[]
	'''
	for i in range(len(p)):
		#print('i = ',i)
		for j in p.iloc[i]['neighbors']:
			#print(j)
			#if type(j)!=int: #se uso MondrianPolygon è pieno di nan
			#	continue
			'''
			if tagli_paralleli == True:
				data1 = a.query('part_number=='+str(p.iloc[i]['part_number']))
				data2 = a.query('part_number=='+str(j))
			else:
			'''	
			data1 = pd.DataFrame(m_leaf[i])#pd.DataFrame(m[p.iloc[i]['part_number']]) #m[p.iloc[i]['part_number']][2]
			data2 = pd.DataFrame(m_leaf[p[p['part_number']==j].index[0]])#pd.DataFrame(m[j]) #m[j][2]
				
 
			part1.append(p.iloc[i]['part_number'])
			part2.append(j)
			
			if score == 'var':
				var_part_unica,var_entrambe_separate = calcolo_varianza_part_vicine(data1,data2)
				v_unica.append(var_part_unica)
				v_sep.append(var_entrambe_separate)
		
			if score == 'centroid':
				if (len(data1)==0) and (len(data2)==0):
					ratio_centroid.append(np.nan)
					difference_centroid.append(np.nan)
				else:
					ratio,difference  = Centroid_part_vicine(data1,data2)
					ratio_centroid.append(ratio)
					difference_centroid.append(difference)
 
			if score == 'min':
				media_i,min_dist_i,min_data1,min_data2,mean1,mean2 = MinDistBetweenPart(data1,data2)
				differenza_minimi = abs(min_dist_i - media_i) + min_data1 + min_data2
				diff_minimi.append(differenza_minimi)
				#differenza_minimi = abs(min_dist_i*(min_data1/mean1)*(min_data2/mean2) - media_i)
				#3differenza_minimi = (min_dist_i - media_i)*(min_data1/mean1)*(min_data2/mean2)
				#4differenza_minimi = (min_dist_i - media_i)*(min_data1/mean1 + min_data2/mean2)
				#1differenza_minimi = (min_dist_i - media_i)*min_data1*min_data2
				#prova5 = differenza_minimi = (min_dist_i*(min_data1/mean1 + min_data2/mean2) - media_i)
				#prova6 = differenza_minimi = (min_dist_i*(min_data1/mean1)*(min_data2/mean2) - media_i)
				#valoreassoluto = abs(differenza_minimi)
				#14agosto = differenza_minimi = abs(min_dist_i - media_i)*(min_data1/mean1)*(min_data2/mean2)
				
				
				#list_min_dist_i.append(min_dist_i)
				#list_min_data1.append(min_data1)
				#list_min_data2.append(min_data2)
				#list_mean1.append(mean1)
				#list_mean2.append(mean2)
				
				
	if score == 'var':
		part_links = {'part1':part1,'part2':part2,'var_part_unica':v_unica,'var_sep':v_sep}
		part_links = pd.DataFrame(part_links)
		part_links['var_ratio'] = part_links['var_part_unica']/part_links['var_sep']
		
	if score == 'centroid':
		part_links = {'part1':part1,'part2':part2,'ratio_centroid':ratio_centroid,'diff_centroid':difference_centroid}
		part_links = pd.DataFrame(part_links)

	if score == 'min':
		part_links = {'part1':part1,'part2':part2,'diff_min':diff_minimi}#,'min_dist':list_min_dist_i,'min_data1':list_min_data1,'min_data2':list_min_data2,'mean1':list_mean1,'mean2':list_mean2}
		part_links = pd.DataFrame(part_links)


	#df_var['diff_pesata'] = df_var['mean_dist_centr12']/(df_var['#points_1']+df_var['#points_2']) - (df_var['mean_dist_centr1']/df_var['#points_1'] + df_var['mean_dist_centr2']/df_var['#points_2'])/2
	#df_var['mean_difference_neighbors'] = 
	#df_var['diff_pesata'] = df_var['mean_dist_centr12']*(df_var['#points_1']+df_var['#points_2']) - (df_var['mean_dist_centr1']*df_var['#points_1'] + df_var['mean_dist_centr2']*df_var['#points_2'])/2
	part_links = part_links.sort_values(by='diff_min',ascending=True)
	part_links.index = np.arange(len(part_links))
	#part_links = part_links.drop(part_links[0:int(len(part_links)/2)].index*2)
	#part_links.index = np.arange(len(part_links))
	
	return part_links



# weight = 'var_ratio','ratio_centroid','diff_centroid','diff_min'
def Network(part_links,weight):
	
	G = nx.Graph()
	for i in range(len(part_links)):
		G.add_edge(part_links['part1'].iloc[i],part_links['part2'].iloc[i],weight=part_links[weight].iloc[i])
	
	A = nx.to_pandas_adjacency(G)
	edgelist = nx.to_pandas_edgelist(G)
	edgelist = edgelist.sort_values(by='weight',ascending=False)
	edgelist.index = np.arange(len(edgelist))
	
	return G,A,edgelist


#A.to_csv(name+'_ad_matrix.txt',sep='\t',index=True)



def Percolation(X,G,edgelist,tagli_paralleli,merged_part,p,m_leaf): 
	
	number_connected_components = []
	connected_components = []
	list_class = []
	
	for i in range(len(edgelist)):
		#print(i)
		G.remove_edge(edgelist['source'].iloc[i],edgelist['target'].iloc[i])
		number_connected_components.append(nx.number_connected_components(G))
		conn_comp = [list(ele) for ele in list(nx.connected_components(G))]
		conn_comp =[[int(s) for s in sublist] for sublist in conn_comp]
		
		for j in range(len(conn_comp)):
			for k in conn_comp[j]:
				if k in list(merged_part['part_to_merge']):
					#print(k)
					associated_part = merged_part[merged_part['part_to_merge']==k]['part_single_data']
					for l in range(len(associated_part)):
						conn_comp[j].append(float(associated_part.iloc[l]))
		
		connected_components.append(conn_comp)
		
		data = AssignClass(G,X,p,m_leaf,tagli_paralleli)#AssignClass(G,X,part,m,tagli_paralleli)
		data['index'] = data['index'].astype(int)
		list_class.append(data[['index','class']])
	
	edgelist_percolation = edgelist.copy()
	edgelist_percolation['number_conn_comp'] = number_connected_components	
	edgelist_percolation['conn_comp'] = connected_components 
	edgelist_percolation['list_class'] = list_class
	#print(edgelist_percolation[edgelist.columns[:-1]])
	#print(connected_components)
	
	list_class_fin = []
	conn_comp_fin = []
	list_class_fin.append(edgelist_percolation['list_class'].iloc[0])
	conn_comp_fin.append([list(ele) for ele in list(edgelist_percolation['conn_comp'].iloc[0])])
	for i in range(1,len(edgelist_percolation)):
		if edgelist_percolation['number_conn_comp'].iloc[i] != edgelist_percolation['number_conn_comp'].iloc[i-1]:
			list_class_fin.append(edgelist_percolation['list_class'].iloc[i])
			#conn_comp_fin.append([list(ele) for ele in list(edgelist_percolation['conn_comp'].iloc[i])])
			conn_comp_fin.append(edgelist_percolation['conn_comp'].iloc[i])
			

	return edgelist_percolation,list_class_fin,conn_comp_fin






def Classification_TD(part,m,X,namefile,score,weight,tagli_paralleli):	
	
	#part_copy = part.copy()
	#m_copy = m.copy()	
	merged_part,p,m_leaf =  MergePart(m,part)
	print(merged_part)
	part_links = PartLinkScore(X,p,m_leaf,score,tagli_paralleli)
	
	G,A,edgelist = Network(part_links,weight)
	A.to_csv(namefile+'_ad_matrix.txt',sep='\t',index=True)
	
	edgelist_percolation,list_class_fin,conn_comp_fin = Percolation(X,G,edgelist,tagli_paralleli,merged_part,p,m_leaf)
	
	with open(namefile+'_list_class.json', 'w') as f:
	    f.write(json.dumps([df.to_dict() for df in list_class_fin]))
	
	with open(namefile+'_conn_comp.json', 'w') as f:
	    f.write(json.dumps([df for df in conn_comp_fin]))

	return




def Classification_BU(m,part,weight,score):#,namefile):
	
	m_leaf,p =  MergePart_SingleData(m,part)
	print(p)
	G = nx.Graph()
	for i in range(len(p)):
		G.add_node(p['part_number'].iloc[i])
	
	list_p = []
	list_m = []
	list_p.append(p)
	list_m.append(m_leaf)
	while nx.number_connected_components(G) > 1:
		part_links = PartLinkScore(m_leaf,p,score)
		G.add_edge(part_links['part1'].iloc[0],part_links['part2'].iloc[0],weight=part_links[weight].iloc[0])
		m_leaf,p = MergePart(m_leaf,p,part_links['part1'].iloc[0],part_links['part2'].iloc[0])
		list_p.append(p)
		list_m.append(m_leaf)
		print(p)
	
	#list_p.reverse()
	#list_m.reverse()
		
#	with open(namefile+'_p.json', 'w') as f:
#	    f.write(json.dumps([df.to_dict() for df in list_p]))
	
#	with open(namefile+'_m_leaf.json', 'w') as f:
#	    f.write(json.dumps([df for df in list_m]))

		
	return list_m,list_p



	 	

def Plot2D(part,list_m,list_p,number_of_clusters):

	p = list_p[number_of_clusters-1]
	for i in range(len(p)):
		p['merged_part'].iloc[i].append(p['part_number'].iloc[i])

	#sns.set_style('whitegrid')
	fig,ax = plt.subplots()
		
	color=cm.rainbow(np.linspace(0,1,len(p)))
	for i in range(len(p)):
		for j in p['merged_part'].iloc[i]:
			box = part[part['part_number']==j]['box'][0]
			pol = Polygon(box, facecolor=color[i], alpha=0.5, edgecolor='black')
			ax.add_patch(pol)
			
			b = pd.DataFrame(box)
			x_avg = np.mean(b[0])
			y_avg = np.mean(b[1])
			ax.text(x_avg,y_avg,j)
		
		data = pd.DataFrame(list_m[0][0])
		ax.scatter(data['0'],data['1'],s=10,alpha=0.5,color='b')
		
	plt.show()
	return
	





def ClassificationScore(list_class,name_file):
	
	pair = list(combinations(np.arange(len(list_class)),2))
	
	coeff_tot = []
	for k in range(len(pair)):
	
		coeff=[]
		index1 = pair[k][0]
		index2 = pair[k][1]
		for i in range(min(len(list_class[index1]),len(list_class[index2]))):
			cl1 = pd.DataFrame(list_class[index1][i])
			cl1.columns = ['index','class1']
			cl2 = pd.DataFrame(list_class[index2][i])
			cl2.columns = ['index','class2']
			
			cl = pd.merge(cl1,cl2,right_on='index',left_on='index')
			coeff.append(adjusted_mutual_info_score(cl['class1'],cl['class2']))
			
		coeff_tot.append(coeff)
	
	coeff_medio = pd.DataFrame(coeff_tot).mean()
	
	fig,ax = plt.subplots()
	ax.plot(np.arange(2,len(coeff_medio)+1),coeff_medio[1:])
	ax.scatter(np.arange(2,len(coeff_medio)+1),coeff_medio[1:])
	for i in range(len(coeff_tot)):
		ax.plot(np.arange(2,len(coeff_tot[i])+1),coeff_tot[i][1:],alpha=0.2)
	plt.show()
	
	if name_file != False:
		plt.savefig(name_file)
	
	fig,ax = plt.subplots()
	ax.plot(np.arange(2,len(coeff_medio)+1),coeff_medio[1:])
	ax.scatter(np.arange(2,len(coeff_medio)+1),coeff_medio[1:])
	plt.show()
	#linestyle = '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
	
	if name_file != False:
		plt.savefig(name_file+'_medio')
	
	return coeff_medio
	
	
	



