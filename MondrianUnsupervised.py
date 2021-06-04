import numpy as np
from numpy.random import choice
import pandas as pd


#from sklearn.metrics import accuracy_score
from itertools import combinations,combinations_with_replacement	
import networkx as nx
from scipy.spatial.distance import pdist






def Cut_distanza(data):
	
	
	dist_matrix = pd.DataFrame(squareform(pdist(data)))
	dist_matrix.max()
	


	return




def Cut(data,d):
	
	
	intervals = pd.DataFrame()

	for dim in d:
			
			
		
		if data[dim].max() == data[dim].min():
			return
		
		
		data_ordered =  data[dim].sort_values()
		#data_ordered.index = np.arange(len(data_ordered))
		data_ordered_min = data_ordered[:-1]
		data_ordered_max = data_ordered[1:]
		data_ordered_min.index = np.arange(len(data_ordered_min))
		data_ordered_max.index = np.arange(len(data_ordered_max))
		data_interval = pd.merge(data_ordered_min,data_ordered_max, left_index=True, right_index=True)
		data_interval.columns = ['min','max']
		
		data_interval['interval'] = data_interval['max'] - data_interval['min']
		intervallo_prescelto = data_interval[data_interval['interval']==data_interval['interval'].max()].copy()
		intervallo_prescelto['dim'] =  dim
		intervals = pd.concat([intervals,intervallo_prescelto])
		
		
	intervals.index = np.arange(len(d))
	
	
	p = intervals['interval']/intervals['interval'].sum()
	d_cut = choice(intervals['dim'],p=p,replace=False)

	distance = intervals['interval'].iloc[d_cut]
	x = intervals['max'].iloc[d_cut] - distance/2	
	
	
	return d_cut,x,distance







def calcolo_varianza(data,d_cut,x):
	
	data1 = data[data[d_cut]>x]
	data2 = data[data[d_cut]<x]
	
	if (len(data1)!=1) & (len(data2)!=1):
		
	
		pd1 = pdist(data1)
		pd2 = pdist(data2)
		
		pd = pdist(data)
		pd12 = np.hstack([pd1, pd2])
		
		##############var(pd)/var(pd12)
		
		
		score_1 = np.abs(np.log(np.var(pd12)/np.var(pd)))
		
		score_2 = np.abs(np.log(np.var(pd1)/np.var(pd2)))
		
		s = [score_1,score_2]
		
		return s 
		
	
		
	else:
		s='nan'
		return s







def Cut_confronto_split_intervalli(data,d):
	
	
	intervals = pd.DataFrame()

	for dim in d:
			
			
		
		if data[dim].max() == data[dim].min():
			return
		
		
		data_ordered =  data[dim].sort_values()
		#data_ordered.index = np.arange(len(data_ordered))
		data_ordered_min = data_ordered[:-1]
		data_ordered_max = data_ordered[1:]
		data_ordered_min.index = np.arange(len(data_ordered_min))
		data_ordered_max.index = np.arange(len(data_ordered_max))
		data_interval = pd.merge(data_ordered_min,data_ordered_max, left_index=True, right_index=True)
		
		data_interval.columns = ['min','max']
		data_interval['interval'] = data_interval['max'] - data_interval['min']
		data_interval.loc[data_interval.index,'dim'] = dim
		
		intervals = pd.concat([intervals,data_interval])
		
	intervals = intervals.sort_values(by='interval',ascending=False)
	intervals.index = np.arange(len(intervals))

	c=0
	for i in range(len(intervals)):
		c += 1
		print(c)
		
		#p = intervals['interval']/intervals['interval'].sum()
		#d_cut = choice(intervals['dim'],p=p,replace=False)
		
		distance = intervals['interval'].iloc[i]
		x = intervals['max'].iloc[i] - distance/2 
		d_cut = intervals['dim'].iloc[i]
		
		s = calcolo_varianza(data,d_cut,x)

		
		
		
		if (len(s)==2) & (s[0]>s[1]):
			return d_cut,x,distance
	
		
	

	

	return 


		





def MondrianUnsupervised_SingleCut(data,t0,l,lifetime,father):


	# array di lunghezze intervalli 
	ld = []
	for i in l:
		ld.append(i[1]-i[0])
		
		
	# linear dimension
	LD = sum(ld)
	
	
	# dimensioni
	d = np.arange(len(l))

	# considero dati solo nell'intervallo l
	for i in d:
		data = data[(data[i]>l[i][0]) & (data[i]<l[i][1])]   



	# genera tempo per cut
	time_cut = np.random.exponential(1/LD) 
	t0 += time_cut
	
	
	if t0 > lifetime:
		return
	
	#d_cut,x,distance = Cut(data,d)
	d_cut,x,distance = Cut_confronto_split_intervalli(data,d)
	#print(d_cut,x)				
				
	l_min = l.copy()
	l_max = l.copy()
	l_min[int(d_cut)] = [l[int(d_cut)][0],x]
	l_max[int(d_cut)] = [x,l[int(d_cut)][1]]
	
	
	
	# metti qui calcolo varianza
	#score_1,score_2 = calcolo_varianza(data,d_cut,x)  
	
	
	risultato1 = [t0, l_min]
	risultato2 = [t0, l_max]
				
				
			
				
				
	risultato = [risultato1, risultato2, distance , t0, d_cut, father, x]
	
	


		
	return risultato
	
		











def MondrianUnsupervised(X,t0,lifetime): 
	
	
	
	data = pd.DataFrame(X)
	
	
	spazio_iniziale = []
	for i in range(len(X[0])):
		length = data[i].max() - data[i].min()
		spazio_iniziale.append([data[i].min() - length*0.05,data[i].max() + length*0.05])
	

	m=[]
	count_part_number = 0
	m0 = [ t0,spazio_iniziale,count_part_number ] 
	m.append(m0)
	
	
	box = []
	distance = []
	time = []
	dim = []
	x = []
		
	father = []
	part_number = []
	
	
	box.append(np.reshape(spazio_iniziale,(1,len(spazio_iniziale)*2))[0])
	distance.append('nan')
	dim.append('nan')
	x.append('nan')		
	time.append(t0)
	father.append('nan')
	part_number.append(count_part_number)
	
			
			
		
	
	
	for i in m:

	
		try:
			

			 
		 
			mondrian = MondrianUnsupervised_SingleCut(data,i[0],i[1],lifetime,i[2])
			
			m.append([mondrian[0][0],mondrian[0][1],count_part_number+1])
			count_part_number += 1
			part_number.append(count_part_number)
			
			m.append([mondrian[1][0],mondrian[1][1],count_part_number+1])
			count_part_number += 1
			part_number.append(count_part_number)
			
			
	
		
			for j in range(2):
				box.append(np.reshape(mondrian[j][1],(1,len(spazio_iniziale)*2))[0])
				distance.append(mondrian[2])
				time.append(mondrian[3])
				dim.append(mondrian[4])
				father.append(mondrian[5])
				x.append(mondrian[6])
				



		except  TypeError:
			continue
		
		
	
	
	names = []
	for i in range(len(spazio_iniziale)):
		for j in ['min','max']:
			names.append(j+str(i))
	
	
	df_box = pd.DataFrame(box)
	df_box.columns = names	
	df = {'time':time,'father':father,'part_number':part_number,'dim':dim,'distance':distance,'x':x}
	df = pd.DataFrame(df)
	#df_part.loc[ (df_part['part_number'] not in df_part['father']==True),'leaf'] = True
	#df_part.loc[*[(df_part['part_number'].iloc[i] not in df_part['father']) for i in range(len(df_part))]]	=True	


	leaf = []
	for i in range(len(df)):
		if df['part_number'].iloc[i] not in df['father'].unique():
			leaf.append(True)
		else:
			leaf.append(False)
		
			
		
	df['leaf'] = leaf
		
	
	part = pd.merge(df, df_box, left_index=True, right_index=True)



	return part










  





def Graph(part):


	G = nx.Graph()
	G.add_node(0)
	for i in range(1,len(part)):
		G.add_edge(part['father'].iloc[i],part['part_number'].iloc[i],weight=part['distance'].iloc[i])
	
	#fig,ax = plt.subplots()
	#nx.draw(G,with_labels=True)
	#plt.show()
	
	return G
	

# calcola shortest_path_length
# link con peso maggiore allontanano due nodi





def ShortestDistance(part):
	
	G = Graph(part)
	
		
		
	node_pair = list(combinations_with_replacement(part.query('leaf==True')['part_number'], 2))	
	
	shortest_path = []
	for i in node_pair:
		source = i[0]
		target = i[1]
		shortest_path.append(nx.shortest_path_length(G,source,target,weight='weight'))
		
	
	dist = pd.DataFrame(node_pair)
	dist.columns = ['source_node','target_node']
	dist['shortest_path_length'] = shortest_path
	
	
	return dist











# associa ad ogni punto la partizione a cui appartiene

def AssignPartition(X,part):	
	
	
	
	d = np.arange(len(X[0]))
	
	
	part_number = []

	for i in X:
		p = part.query('leaf==True').copy()
		
		for j in d:
			p.loc[p.index,'data'+str(j)] = i[j]
			p = p.query("(data"+str(j)+">min"+str(j)+") & (data"+str(j)+"<max"+str(j)+")").copy()
		
		part_number.append(p['part_number'].iloc[0])



	X = pd.DataFrame(X)
	X['part_number'] = part_number
	
		
	return X












def MondrianIterator(number_iterations,X,t0,lifetime):

	
	# calcolo ogni possibile coppia di punti
	pair_index = list(combinations(np.arange(len(X)), 2))
	pair_index = pd.DataFrame(pair_index)
	pair_index.columns = ['index1','index2']
	
	
	
	
	
	for i in range(number_iterations):
		
		print(i+1)
		
		part = MondrianUnsupervised(X,t0,lifetime)
		# calcolo distanze fra partizioni
		dist = ShortestDistance(part)
		
		
		#associo ogni punto a una partizione e gli assegno un indice
		data = AssignPartition(X,part)
		data['index'] = data.index
		
		
		# associo ad ogni coppia la partizione del primo punto
		df = pd.merge(pair_index,data, how='left', left_on='index1', right_on='index')
		df = df[['index1','index2','part_number']]
		df.columns = ['index1','index2','part_number1']
		
		#associo ad ogni coppia la partizione del secondo pounto
		df = pd.merge(df,data, how='left', left_on='index2', right_on='index')
		df = df[['index1','index2','part_number1','part_number']]
		df.columns = ['index1','index2','part_number1','part_number2']
		
		# info della combinazioni delle due partizioni in unica colonna
		df.loc[df.index,'part_pair'] = [*[(str(df[['part_number1','part_number2']].iloc[s].min())+'+'+str(df[['part_number1','part_number2']].iloc[s].max())) for s in df.index]]
		dist.loc[dist.index,'part_pair'] = [*[(str(dist[['source_node','target_node']].iloc[s].min())+'+'+str(dist[['source_node','target_node']].iloc[s].max())) for s in dist.index]]


		dist_points = pd.merge(df,dist, how='left',left_on='part_pair', right_on='part_pair').copy()
		dist_points = dist_points[['index1', 'index2', 'part_number1', 'part_number2', 'shortest_path_length']]

		pair_index['min_path_dist'+str(i)] = dist_points['shortest_path_length']
		
		
		
	pair_index.loc[pair_index.index,'avg_dist'] = [*[np.mean(list(pair_index.iloc[s])[2:]) for s in pair_index.index]]
	
	
	
	#coppia di punti con media della distanza
	pair_with_avg = pair_index[['index1','index2','avg_dist']]
	
	
	
	G = nx.Graph()
	for i in range(len(pair_with_avg)):
		G.add_edge(pair_with_avg['index1'].iloc[i],pair_with_avg['index2'].iloc[i],weight=pair_with_avg['avg_dist'].iloc[i])

	
	
	matrix = nx.to_pandas_adjacency(G)
	
	data.drop('part_number',axis=1)
	
	


	
	return matrix,data


