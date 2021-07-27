

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist,pdist
import networkx as nx



def calcolo_varianza_part_vicine(data1,data2):
	

	data1 = data1.drop('index',axis=1)
	data2 = data2.drop('index',axis=1)
	
	pd1 = pdist(data1)
	pd2 = pdist(data2)
	
	data = np.vstack([np.array(data1),np.array(data2)])
	pd = pdist(data)
	pd12 = np.hstack([pd1, pd2])
	
	var_part_unica = np.var(pd)
	var1 = np.var(pd1)
	var2 = np.var(pd2)
	var_entrambe_separate = np.var(pd12) 


	return var_part_unica,var1,var2,var_entrambe_separate




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
	
	mean_dist12 = np.mean(dist[2])
	mean_dist1 = np.mean(dist[0])
	mean_dist2 = np.mean(dist[1])	
	ratio = np.mean(dist[2])/np.mean(np.vstack([dist[0],dist[1]]))	
	difference = np.mean(dist[2]) - np.mean(np.vstack([dist[0],dist[1]]))
	
	data1_number = len(data1)
	data2_number = len(data2)


	return ratio,difference,mean_dist1,mean_dist2,mean_dist12,data1_number,data2_number	  







def calcolo_dist_part_vicine(data1,data2):
	
	data1 = data1.drop('index',axis=1)
	data2 = data2.drop('index',axis=1)
	
	pd1 = cdist(data1,data1)
	i1_data1,i2_data1 = np.tril_indices(len(data1), k=-1)
	pd2 = cdist(data2,data2)
	i1_data2, i2_data2 = np.tril_indices(len(data2), k=-1)


	df1 = {'i1':i1_data1,'i2':i2_data1,'dist':pd1[i1_data1,i2_data1]}
	df1 = pd.DataFrame(df1)
	min1 = []
	for i in range(df1['i1'].max()+1):
		x = df1.query('i1=='+str(i)+' or i2=='+str(i))['dist'].min()
		min1.append(x)


	df2 = {'i1':i1_data2,'i2':i2_data2,'dist':pd2[i1_data2,i2_data2]}
	df2 = pd.DataFrame(df2)
	min2 = []
	for i in range(df2['i1'].max()+1):
		x = df2.query('i1=='+str(i)+' or i2=='+str(i))['dist'].min()
		min2.append(x)
	
	
	min_tot = np.hstack([min1,min2])
	#media1 = np.mean(min1)
	#media2 = np.mean(min2)
	#min_media_sep = (media1+media2)/2
	media = np.mean(min_tot)
	var = np.sqrt(np.var(min_tot))


	
	dist = cdist(data1,data2)
	min_dist_fra_partizioni = dist.min()

	
	return media,var,min_dist_fra_partizioni	  












part1 = []
part2 = []
v_unica = []
v_sep = []
ratio_centroid = []
difference_centroid = []
v1 = []
v2= []

mean_centr1 = []
mean_centr2 = []
mean_centr12 = []
number_of_points1 = []
number_of_points2 = []

diff_minimi = []

p = part.query('leaf==True').copy()
p.index = np.arange(len(p))


for i in range(len(p)):
	print(i)
	
	for j in p.iloc[i]['neighbors']:
		if type(j)!=int:
			continue
		
		#per griglia regolare
		#data = 
		#data1 = data[(data[0]>p['min0'].iloc[i]) & (data[0]<p['max0'].iloc[i]) & (data[1]>p['min1'].iloc[i]) & (data[1]<p['max1'].iloc[i])]
		#data2 = data[(data[0]>p.query('part_number=='+str(j))['min0'].iloc[0]) & (data[0]<p.query('part_number=='+str(j))['max0'].iloc[0]) & (data[1]>p.query('part_number=='+str(j))['min1'].iloc[0]) & (data[1]<p.query('part_number=='+str(j))['max1'].iloc[0])]
		
		#per solito metodo
		data1 = m[p.iloc[i]['part_number']][2]
		data2 = m[j][2]
		
		if (len(data1)==1) or (len(data2)==1):
			continue
		
		
		part1.append(p.iloc[i]['part_number'])
		part2.append(j)
		
		var_part_unica,var1,var2,var_entrambe_separate = calcolo_varianza_part_vicine(data1,data2)
		v_unica.append(var_part_unica)
		v_sep.append(var_entrambe_separate)
		v1.append(var1)
		v2.append(var2)
		
		
		if (len(data1)==0) and (len(data2)==0):
			ratio_centroid.append(np.nan)
			difference_centroid.append(np.nan)
		else:
			ratio,difference,mean_dist1,mean_dist2,mean_dist12,data1_number,data2_number  = Centroid_part_vicine(data1,data2)
			ratio_centroid.append(ratio)
			difference_centroid.append(difference)
			mean_centr1.append(mean_dist1)
			mean_centr2.append(mean_dist2)
			mean_centr12.append(mean_dist12)
			number_of_points1.append(data1_number)
			number_of_points2.append(data2_number)
			
			
		media_i,std_dev_i,min_dist_i = calcolo_dist_part_vicine(data1,data2)
		#media.append(media_i)
		#std_dev.append(std_dev_i)
		#min_dist.append(min_dist_i)
		differenza_minimi = min_dist_i - media_i
		diff_minimi.append(differenza_minimi)



#varianza
df_var = {'part1':part1,'part2':part2,'#points_1':number_of_points1,'#points_2':number_of_points2,'var1':v1,'var2':v2,'var_part_unica':v_unica,'var_sep':v_sep,'ratio_centroid':ratio_centroid,'difference_centroid':difference_centroid,'mean_dist_centr1':mean_centr1,'mean_dist_centr2':mean_centr2,'mean_dist_centr12':mean_centr12,'diff_minimi':diff_minimi}
df_var = pd.DataFrame(df_var)
df_var['ratio'] = df_var['var_part_unica']/df_var['var_sep']


#df_var['diff_pesata'] = df_var['mean_dist_centr12']/(df_var['#points_1']+df_var['#points_2']) - (df_var['mean_dist_centr1']/df_var['#points_1'] + df_var['mean_dist_centr2']/df_var['#points_2'])/2
#df_var['mean_difference_neighbors'] = 
#df_var['diff_pesata'] = df_var['mean_dist_centr12']*(df_var['#points_1']+df_var['#points_2']) - (df_var['mean_dist_centr1']*df_var['#points_1'] + df_var['mean_dist_centr2']*df_var['#points_2'])/2


#df_var =df_var.dropna()

#df_var = df_var.sort_values(by='difference_centroid',ascending=False)
#df_var.index = np.arange(len(df_var))


# network

G = nx.Graph()
for i in range(len(df_var)):
	G.add_edge(df_var['part1'].iloc[i],df_var['part2'].iloc[i],weight=df_var['diff_minimi'].iloc[i])


A = nx.to_pandas_adjacency(G)
edgelist = nx.to_pandas_edgelist(G)


#connsidero degree
degree = pd.DataFrame(nx.degree(G,weight='weight'))
degree.columns = ['node','degree']

#edgelist = pd.merge()


edgelist = edgelist.sort_values(by='weight',ascending=False)
edgelist.index = np.arange(len(edgelist))

A_nuovo = []

number_connected_components = []
connected_components = []
#number_connected_components.append(nx.number_connected_components(G))
for i in range(len(edgelist)):
	G.remove_edge(edgelist['source'].iloc[i],edgelist['target'].iloc[i])
	number_connected_components.append(nx.number_connected_components(G))
	A_nuovo.append(nx.to_pandas_adjacency(G))
	connected_components.append(list(nx.connected_components(G)))
	
edgelist['number_conn_comp'] = number_connected_components	
#edgelist['conn_comp'] = connected_components

print(edgelist[edgelist.columns[:-1]])
print(connected_components)








#%%

#part[['time', 'father', 'part_number', 'neighbors', 'leaf']]
#part.query('leaf==True')
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist,pdist
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.patches import Polygon



def calcolo_varianza_part_vicine(data1,data2):
	
	data1 = data1.drop('index',axis=1)
	data2 = data2.drop('index',axis=1)
	
	pd1 = pdist(data1)
	pd2 = pdist(data2)
	
	data = np.vstack([np.array(data1),np.array(data2)])
	pd = pdist(data)
	pd12 = np.hstack([pd1, pd2])
	
	var_part_unica = np.var(pd)
	var1 = np.var(pd1)
	var2 = np.var(pd2)
	var_entrambe_separate = np.var(pd12) 

	
	return var_part_unica,var1,var2,var_entrambe_separate


def calcolo_dist_part_vicine(data1,data2):
	
	data1 = data1.drop('index',axis=1)
	data2 = data2.drop('index',axis=1)
	
	pd1 = cdist(data1,data1)
	i1_data1,i2_data1 = np.tril_indices(len(data1), k=-1)
	pd2 = cdist(data2,data2)
	i1_data2, i2_data2 = np.tril_indices(len(data2), k=-1)


	df1 = {'i1':i1_data1,'i2':i2_data1,'dist':pd1[i1_data1,i2_data1]}
	df1 = pd.DataFrame(df1)
	min1 = []
	for i in range(df1['i1'].max()+1):
		x = df1.query('i1=='+str(i)+' or i2=='+str(i))['dist'].min()
		min1.append(x)


	df2 = {'i1':i1_data2,'i2':i2_data2,'dist':pd2[i1_data2,i2_data2]}
	df2 = pd.DataFrame(df2)
	min2 = []
	for i in range(df2['i1'].max()+1):
		x = df2.query('i1=='+str(i)+' or i2=='+str(i))['dist'].min()
		min2.append(x)
	
	
	min_tot = np.hstack([min1,min2])
	#media1 = np.mean(min1)
	#media2 = np.mean(min2)
	#min_media_sep = (media1+media2)/2
	media = np.mean(min_tot)
	var = np.sqrt(np.var(min_tot))


	
	dist = cdist(data1,data2)
	min_dist_fra_partizioni = dist.min()

	
	return media,var,min_dist_fra_partizioni





#%%

#devi fare la media delle distanze minime fra coppie di punti


#varianza
soglia= 2.5
#minimi
coeff=3





part1 = []
part2 = []
v_unica = []
#v1 = []
#v2 = []
v_sep = []
media=[]
std_dev=[]
min_dist=[]

p = part.query('leaf==True').copy()
p.index = np.arange(len(p))

for i in range(len(p)):
	print(i)
	
	for j in p.iloc[i]['neighbors']:
		if type(j)!=int:
			continue
		
		
		data1 = m[p.iloc[i]['part_number']][2]
		data2 = m[j][2]
		
		if (len(data1)==1) or (len(data2)==1):
			continue
		
		 
		part1.append(p.iloc[i]['part_number'])
		part2.append(j)
		
		var_part_unica,var1,var2,var_entrambe_separate = calcolo_varianza_part_vicine(data1,data2)
		v_unica.append(var_part_unica)
		#v1.append(var1)
		#v2.append(var2)
		v_sep.append(var_entrambe_separate)

		media_i,std_dev_i,min_dist_i = calcolo_dist_part_vicine(data1,data2)
		media.append(media_i)
		std_dev.append(std_dev_i)
		min_dist.append(min_dist_i)


#varianza		
df_var = {'part1':part1,'part2':part2,'var_part_unica':v_unica,'var_sep':v_sep}
df_var = pd.DataFrame(df_var)
df_var['ratio'] = df_var['var_part_unica']/df_var['var_sep']

df_var.eval('stessa_classe = ratio<'+str(soglia),inplace=True)



#minimi ecc
df_min = {'part1':part1,'part2':part2,'media':media,'std_dev':std_dev,'min_dist':min_dist}
df_min = pd.DataFrame(df_min)

df_min.eval('stessa_classe = min_dist < media + '+str(coeff)+'*std_dev',inplace=True)  #puoi moltiplicare qualcosa per la std dev


df = [df_var,df_min]
titolo = ['calcolo varianza, soglia='+str(soglia),'min, coeff='+str(coeff)]


dataframe_tot = []





for k in range(2):
	partizione = []
	stessa_classe = []
	classe_diversa = []
	for i in p['part_number']:
		partizione.append(i)
		
		df_ridotto1 = df[k].query('part1=='+str(i)+' and stessa_classe==True').copy()
		df_ridotto2 = df[k].query('part2=='+str(i)+' and stessa_classe==True').copy()
		
		part_stessa_classe = np.unique(np.hstack([list(df_ridotto1['part2']),list(df_ridotto2['part1'])]))
		stessa_classe.append(part_stessa_classe)
		
		df_ridotto1 = df[k].query('part1=='+str(i)+' and stessa_classe==False').copy()
		df_ridotto2 = df[k].query('part2=='+str(i)+' and stessa_classe==False').copy()
		
		part_classe_diversa = np.unique(np.hstack([list(df_ridotto1['part2']),list(df_ridotto2['part1'])]))
		classe_diversa.append(part_classe_diversa)
		
	dataframe = {'partizione':partizione,'stessa_classe':stessa_classe,'classe_diversa':classe_diversa}
	dataframe=pd.DataFrame(dataframe)
		
	
	


	# assegna classe
	
	classe = []
	classe.append([0])
	for i in range(len(dataframe)-1):
		classe.append([])
		
		
	dataframe['classe'] = classe
	#dataframe['classe2'] = classe2
	
	
	h=[]
	for i in range(len(dataframe)):
		if len(dataframe['stessa_classe'].iloc[i]) != 0:
			h.append(dataframe['partizione'].iloc[i])
			break
	for i in h:
		for j in dataframe.query('partizione=='+str(i))['stessa_classe'].iloc[0]:
			dataframe.query('partizione=='+str(j))['classe'].iloc[0].append(0)
			if j not in h:
				h.append(j)
			

	'''
	h = []
	classe = []
	for i in range(len(dataframe)):
		classe.append([])
	for i in range(len(dataframe)):
		if len(dataframe['stessa_classe'].iloc[i]) != 0:
			classe[i].append(1)
			h.append(dataframe['partizione'].iloc[i])
			break
	for i in h:
		for j in dataframe.query('partizione=='+str(i))['stessa_classe'].iloc[0]:
			classe[dataframe[dataframe['partizione'] == j].index[0]].append(1)
			if j not in h:
				h.append(j)
	
	k= []
	for i in range(len(dataframe)):
		if len(classe[i]) == 0:
			classe[i].append(0)  # problema
			k.append(dataframe['partizione'].iloc[i])
			
	for i in k:
		for j in dataframe.query('partizione=='+str(i))['stessa_classe'].iloc[0]:
			classe[dataframe[dataframe['partizione'] == j].index[0]].append(0)
			if j not in k:
				k.append(j)
			
	'''			

	
	
	
	dataframe_tot.append(dataframe)
	
	
	sns.set_style('whitegrid')
	fig,ax = plt.subplots()
	
	
	for i in range(len(p)):
		box_new = p['box'].iloc[i]
		if len(dataframe[dataframe['partizione']==p['part_number'].iloc[i]]['classe'].iloc[0]) !=0:
			poligono = Polygon(box_new, facecolor = 'blue', alpha=0.5, edgecolor='black')
			ax.add_patch(poligono)
			b = pd.DataFrame(box_new)
			x_avg = np.mean(b[0])
			y_avg = np.mean(b[1])
			ax.text(x_avg,y_avg,p['part_number'].iloc[i])#,color='b')
	
		else:
			poligono = Polygon(box_new, facecolor = 'orange',alpha=0.5, edgecolor='black')
			ax.add_patch(poligono)
			b = pd.DataFrame(box_new)
			x_avg = np.mean(b[0])
			y_avg = np.mean(b[1])
			ax.text(x_avg,y_avg,p['part_number'].iloc[i])#,color='orange')
			
	ax.set_title(titolo[k])
				
	ax.scatter(X[:,0],X[:,1],color='b',s=10,alpha=0.5)
				
	plt.show()
			
		
	
	















#%%
#       trovare partizioni vicine tagli perpendicolari



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





#  calcolo varianza per partizioni vicine 


def calcolo_varianza_part_vicine(data,i,j):
	
	
	
	data1 = data.query('part_number=='+str(i))
	data2 = data.query('part_number=='+str(j))
	
	pd1 = pdist(data1)
	pd2 = pdist(data2)
	
	pd = pdist(data)
	pd12 = np.hstack([pd1, pd2])
	
	var_part_unica = np.var(pd)
	var1 = np.var(pd1)
	var2 = np.var(pd2)
	
	#var_part_separate = np.var(pd12)
	#score_1 = np.abs(np.log(np.var(pd12)/np.var(pd)))
	#score_2 = np.abs(np.log(np.var(pd1)/np.var(pd2)))
	#print(score_1>score_2)

	
	return var_part_unica,var1,var2 #score_1,score_2, var_part_unica,var_part_separate





part_vicine = trova_part_vicine(part)
punti = AssignPartition(X,part)

#separazione_corretta = [] 


part1 = []
part2 = []
score_1 = []
score_2 = []
#v_unica = []
#v_sep = []


for i in part_vicine['part_number']:
	
	for j in list(part_vicine[part_vicine['part_number']==i]['neighbors'])[0]:
		part2.append(j)
		part1.append(i)
		
		p = punti.query('(part_number=='+str(i)+') or (part_number=='+str(j)+')').copy()
		
		
		var_part_unica,var1,var2 = calcolo_varianza_part_vicine(punti,i,j)
		#s1,s2,v1,v2 = calcolo_varianza_part_vicine(punti,i,j)
		#score_1.append(s1)
		#score_2.append(s2)
		#v_unica.append(v1)
		#v_sep.append(v2)
		
		
df = {'part1':part1,'part2':part2,'var_part_unica':var_part_unica,'var1':var1,'var2':var2}#'score_1':score_1,'score_2':score_2}		
df = pd.DataFrame(df)		#'v_unica':v_unica,'v_sep':v_sep}#
		#if s1 > s2:
		#	separazione_corretta.append(True)
		#else:
		#	separazione_corretta.append(False)
		

		
		
	
#conto_punti = punti.query('part_number=='+str(i)).count()[0]


