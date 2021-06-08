from scipy.spatial.distance import pdist,squareform,cdist

#%%
def distance_matrix(X):
	
	#associo indice a ciascun punto
	n_d = len(X[0]) # numero dimensioni
	data = pd.DataFrame(X)
	data.columns = [*[f'dim{s}_point' for s in data.columns]]
	data['index'] = data.index
	

	#dataframe di coppie di punti
	pair_points = list(combinations(np.arange(len(X)), 2))
	pair_points = pd.DataFrame(pair_points)
	pair_points.columns = ['index_point_x','index_point_y']
	
	#associo punti alle coppie di indici
	pair_points = pd.merge(pair_points,data, how='left', left_on='index_point_x', right_on='index')
	pair_points = pair_points.drop('index',axis=1)
	pair_points = pd.merge(pair_points,data, how='left', left_on='index_point_y', right_on='index')
	pair_points = pair_points.drop('index',axis=1)
	
	#calcolo distanza fra punti
	pair_points['dist'] = pdist(data.drop('index',axis=1))
	
	
				#coeff angolare
	pair_points = (pair_points.eval('m = (dim1_point_x - dim1_point_y) / (dim0_point_x - dim0_point_y)')
				#intercetta
				.eval('q = (dim0_point_x*dim1_point_y - dim0_point_y*dim1_point_x) / (dim0_point_x - dim0_point_y)'))

	
	#punto medio congiungente i due punti (cut)
	for i in range(n_d):
		pair_points['cut_point'+str(i)] = (pair_points['dim'+str(i)+'_point_x'] + pair_points['dim'+str(i)+'_point_y'])/2
		

	#m_cut
	pair_points = (pair_points.eval('m_cut = -m')
				.eval('q_cut = cut_point1 - m_cut*cut_point0'))
	
	dist_matrix = pd.DataFrame()
	
	for i in range(len(data)):
		
		matrix = pair_points.copy()
		
		for j in data:
			matrix[str(j)] = data[str(j)].iloc[i]
		matrix = matrix.drop(matrix.query('index_point_x==index or index_point_y==index').index)
		matrix = matrix.drop('index',axis=1)
		
		dist_matrix = pd.concat([dist_matrix,matrix])
		
	
	dist_matrix = dist_matrix.eval('q_new_point = dim1_point - m*dim0_point')
	dist_matrix.index = np.arange(len(dist_matrix))
	
	
	#punto intersezione
	dist_matrix['punto_intersez0'] = (dist_matrix['q_new_point'] - dist_matrix['q_cut'])/(2*dist_matrix['m_cut'])
	dist_matrix['punto_intersez1'] = (dist_matrix['q_new_point'] + dist_matrix['q_cut'])/2
	
	#dist_matrix.loc[dist_matrix.index,'dist_new_point'] = [*[np.abs(((dist_matrix['dim1_point'].iloc[i] - dist_matrix['m'].iloc[i]*dist_matrix['dim0_point'].iloc[i] - dist_matrix['q_new_point'].iloc[i])) / np.sqrt(1 + dist_matrix['m'].iloc[i]**2)) for i in range(len(dist_matrix))]]	
	dist_matrix.loc[dist_matrix.index,'pdist_new_point'] = [[*pdist([[dist_matrix['punto_intersez0'].iloc[i],dist_matrix['punto_intersez1'].iloc[i]],[dist_matrix['dim0_point'].iloc[i],dist_matrix['dim1_point'].iloc[i]]])) for i in range(len(dist_matrix))]]


	 
	#dist_matrix = pd.DataFrame(squareform(pdist(data)))
	
#%%




#%%   prova altre distribuzioni di dati
mean1 = (1, 1)
cov1 = [[0.1, 0], [0, 0.1]]
x1 = np.random.multivariate_normal(mean1, cov1, 200)
x1=pd.DataFrame(x1)
x1['cl']=0


mean2 = (4, 4)
cov2 = [[0.1, 0], [0, 0.1]]
x2 = np.random.multivariate_normal(mean2, cov2, 200)
x2=pd.DataFrame(x2)
x2['cl']=1


X = pd.concat([x1,x2])






fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

ax.scatter(x1[0],x1[1])
ax.scatter(x2[0],x2[1])




#per usarlo nelle funzioni
y = np.array(X['cl'])	
X = np.array(X[[0,1]])





#%%



mean1 = (1, 1)
cov1 = [[0.1, 0], [0, 0.1]]
x1 = np.random.multivariate_normal(mean1, cov1, 200)
x1=pd.DataFrame(x1)
x1['cl']=0


mean2 = (4, 4)
cov2 = [[0.1, 0], [0, 0.1]]
x2 = np.random.multivariate_normal(mean2, cov2, 200)
x2=pd.DataFrame(x2)
x2['cl']=1


X = pd.concat([x1,x2])
X.index = np.arange(len(X))
X['index'] = X.index





coppie = list(combinations(np.arange(len(X)), 2))
coppie = pd.DataFrame(coppie)
coppie.columns = ['punto1','punto2']

dist = pd.DataFrame(cdist(X,X))

p1 = []
p2 = []
d = []
for i,j in zip(coppie['punto1'],coppie['punto2']):
	p1.append(i)
	p2.append(j)
	d.append(dist.iloc[i,j])


df = {'punto1':p1,'punto2':p2,'dist':d}
df = pd.DataFrame(df)
	
	
df_dist = pd.merge(df,X[['cl','index']],how='left',left_on='punto1', right_on='index')
df_dist.columns = ['punto1', 'punto2', 'dist', 'cl1', 'index']
df_dist = df_dist.drop('index',axis=1)


df_dist = pd.merge(df_dist,X[['cl','index']],how='left',left_on='punto2', right_on='index')
df_dist.columns = ['punto1', 'punto2', 'dist', 'cl1', 'cl2', 'index']
df_dist = df_dist.drop('index',axis=1)


df_dist['gruppi_separati'] = True
df_dist.loc[df_dist.query('cl1==cl2').index,'gruppi_separati'] = False

df_dist = df_dist[['punto1', 'punto2', 'dist', 'gruppi_separati']]



	




#%% confronto per varianza
	

	#dim = 0
	data_ordered =  data.sort_values(by=0)
	data_ordered = data_ordered.drop(1,axis=1)
		
	data_ordered_min = data_ordered[:-1]
	data_ordered_min.columns = ['min', 'cl_min', 'index_min']
	
	data_ordered_max = data_ordered[1:]
	data_ordered_max.columns = ['max', 'cl_max', 'index_max']
	
	data_ordered_min.index = np.arange(len(data_ordered_min))
	data_ordered_max.index = np.arange(len(data_ordered_max))
	
	data_interval0 = pd.merge(data_ordered_min,data_ordered_max, left_index=True, right_index=True)

	data_interval0.loc[data_interval0.index,'dim'] = 0
		

	#dim = 1
	data_ordered =  data.sort_values(by=1)
	data_ordered = data_ordered.drop(0,axis=1)
		
	data_ordered_min = data_ordered[:-1]
	data_ordered_min.columns = ['min', 'cl_min', 'index_min']
	
	data_ordered_max = data_ordered[1:]
	data_ordered_max.columns = ['max', 'cl_max', 'index_max']
	
	data_ordered_min.index = np.arange(len(data_ordered_min))
	data_ordered_max.index = np.arange(len(data_ordered_max))
	
	data_interval1 = pd.merge(data_ordered_min,data_ordered_max, left_index=True, right_index=True)
	data_interval1.loc[data_interval1.index,'dim'] = 1
		
	
	
	intervals = pd.concat([data_interval0,data_interval1])
	intervals['interval'] = intervals['max'] - intervals['min']
	
	
	intervals['gruppi_separati'] = True
	intervals.loc[intervals.query('cl_min==cl_max').index,'gruppi_separati'] = False

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
		
		
		
	#intervals = intervals.sort_values(by='min')
	intervals.index = np.arange(len(intervals))
	
	intervals['x'] = intervals['max'] - intervals['interval']/2
	
	
	var_ratio = []
	for i in range(len(intervals)):
		d_cut = intervals['dim'].iloc[i]
		x = intervals['x'].iloc[i]
		var = calcolo_varianza(data,d_cut,x)
		var_ratio.append(var)
		
		
	intervals['var_ratio'] = var_ratio
	
	# cancella righe con var_ratio=nan
	intervals = intervals.drop(intervals[intervals['var_ratio']=='nan'].index)
	intervals.index = np.arange(len(intervals))
	


	q=intervals['var_ratio']**50
	p = q/q.sum()
	index_cut = choice(intervals.index,p=p)


	d_cut = int(intervals['dim'].iloc[index_cut])
	distance = intervals['interval'].iloc[index_cut]
	x = intervals['max'].iloc[index_cut] - distance/2


	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
	ax.hist(intervals['var_ratio'])
	plt.show()


	
	return d_cut,x,distance


	


