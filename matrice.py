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
	
	
	
	dist = pd.DataFrame(cdist(X,X))

	p1 = []
	p2 = []
	d = []
	for i,j in zip(pair_points['index_point_x'],pair_points['index_point_y']):
		p1.append(i)
		p2.append(j)
		d.append(dist.iloc[i,j])
	
		
	pair_points = {'index_point_x':p1,'index_point_y':p2,'dist':d}
	pair_points = pd.DataFrame(pair_points)

	
	#associo punti alle coppie di indici
	pair_points = pd.merge(pair_points,data, how='left', left_on='index_point_x', right_on='index')
	pair_points = pair_points.drop('index',axis=1)
	pair_points = pd.merge(pair_points,data, how='left', left_on='index_point_y', right_on='index')
	pair_points = pair_points.drop('index',axis=1)
	

	names = []
	#punto medio congiungente i due punti (cut)
	for i in range(n_d):
		pair_points['cut_point'+str(i)] = (pair_points['dim'+str(i)+'_point_x'] + pair_points['dim'+str(i)+'_point_y'])/2
		#vettore normale al taglio (parallelo a congiungente i due punti)
		pair_points['vettore_'+str(i)] = pair_points['dim'+str(i)+'_point_y'] - pair_points['dim'+str(i)+'_point_x']
		names.append('vettore_'+str(i))
	
	#normalizzo vettore normale
	for i in range(n_d):
		pair_points.loc[pair_points.index,'norm_vect_'+str(i)] = [*[(pair_points['versore_'+str(i)].iloc[j]/np.linalg.norm(pair_points[names].iloc[j])) for j in range(len(pair_points)) ]]
	
	names_norm_vect = []
	names_cut = []
	for i in range(n_d):
		pair_points = pair_points.drop('versore_'+str(i),axis=1)
		names_norm_vect.append('norm_vect_'+str(i))
		names_cut.append('cut_point'+str(i))
		
	#modulo vettore normale
	pair_points.loc[pair_points.index,'magnitude_norm_vect'] = [*[np.dot(pair_points[names_norm_vect].iloc[j],pair_points[names_cut].iloc[j]) for j in range(len(pair_points))]]
	


	dist_matrix = pd.DataFrame()
	
	for i in range(len(data)):
		
		matrix = pair_points.copy()
		
		for j in data:
			matrix[str(j)] = data[str(j)].iloc[i]
		matrix = matrix.drop(matrix.query('index_point_x==index or index_point_y==index').index)
		matrix = matrix.drop('index',axis=1)
		
		dist_matrix = pd.concat([dist_matrix,matrix])
		
	names_point = []	
	for i in range(n_d):
		names_point.append('dim'+str(i)+'_point')
		
	dist_matrix.index = np.arange(len(dist_matrix))
	
	
	#calcolo distanza punti da retta cut
	dist_matrix.loc[dist_matrix.index,'distance_point_cut'] = [*[(np.dot(dist_matrix[names_norm_vect].iloc[j],dist_matrix[names_point].iloc[j]) + dist_matrix['magnitude_norm_vect'].iloc[j]) for j in range(len(dist_matrix))]]
	








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





#%%     confronto varianza





def calcolo_varianza(data,d_cut,x,n):
	
	data1 = data[data[d_cut]>x]
	data2 = data[data[d_cut]<x]
	
	if (len(data1)>=3) & (len(data2)>=3):
		
	
		pd1 = pdist(data1)
		pd2 = pdist(data2)
		
		pd = pdist(data)
		pd12 = np.hstack([pd1, pd2])
		
		var_ratio = np.var(pd)/np.var(pd12) 
		var_ratio = var_ratio**n
		
		return var_ratio 
		
	else:
		s='nan'
		return s








N = [50,100,150,200] # numerosità
n = np.geomspace(0.01, 1, 21) # rumore
pot = [1,5,10,20,30,40,50] #esponente rapporto varianze

numerosità = []
rumore = []
esponente = []
percentuale = []


for a in N:
	print('N: ',a)

	sovrapposizione_classe_diversa = []
	sovrapposizione_stessa_classe = []
	
	
	for b in n:
		print('rumore: ',b)
	 		
		for p in  pot:
			print('esponente: ',p)
			
			
			for rip in range(10):
				print('ripetizione: ',rip+1)
				
				
				mean1 = (0, 0)
				cov1 = [[b, 0], [0, b]]
				x1 = np.random.multivariate_normal(mean1, cov1, a)
				x1=pd.DataFrame(x1)
				x1['cl']=0
				
				
				mean2 = (1, 0)
				cov2 = [[b, 0], [0, b]]
				x2 = np.random.multivariate_normal(mean2, cov2, a)
				x2=pd.DataFrame(x2)
				x2['cl']=1
				
		
		
				
				X = pd.concat([x1,x2])
		
				#X = pd.DataFrame(X)
				#X['cl'] = y
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
				
				
				df_dist = pd.merge(df_dist,X, how='left', left_on='punto1', right_on='index')
				df_dist = df_dist.drop(['cl','index'],axis=1)
				df_dist.columns = ['punto1', 'punto2', 'dist', 'gruppi_separati', 'punto1_0', 'punto1_1']
				df_dist = pd.merge(df_dist,X, how='left', left_on='punto2', right_on='index')
				df_dist = df_dist.drop(['cl','index'],axis=1)
				df_dist.columns = ['punto1', 'punto2', 'dist', 'gruppi_separati', 'punto1_0', 'punto1_1', 'punto2_0', 'punto2_1']
					
				#df1 = df_dist.copy()
				#df2 = df_dist.copy()
				#df1.loc[df1.index,'taglio'] = [*[((df1['punto1_0'].iloc[i]+df1['punto2_0'].iloc[i])/2) for i in range(len(df1))]]
				#df2.loc[df2.index,'taglio'] = [*[((df2['punto1_1'].iloc[i]+df2['punto2_1'].iloc[i])/2) for i in range(len(df2))]]
				#df1['dim'] = 0
				#df2['dim'] = 1
				df_dist.loc[df_dist.index,'taglio'] = [*[((df_dist['punto1_0'].iloc[i]+df_dist['punto2_0'].iloc[i])/2) for i in range(len(df_dist))]]
				#df2.loc[df2.index,'taglio'] = [*[((df2['punto1_1'].iloc[i]+df2['punto2_1'].iloc[i])/2) for i in range(len(df2))]]
				df_dist['dim'] = 0
				
				#dist_finale = pd.concat([df1,df2])
				dist_finale = df_dist.copy()
				
				var_ratio = []
				
				for i in range(len(dist_finale)):
					#print(i)
					d_cut = dist_finale['dim'].iloc[i]
					x = dist_finale['taglio'].iloc[i]
					var = calcolo_varianza(X[[0,1]],d_cut,x,p)
					var_ratio.append(var)
						
						
				dist_finale['var_ratio'] = var_ratio
				
				dist_finale = dist_finale.drop(dist_finale[dist_finale['var_ratio']=='nan'].index)
				
				peso_classe_diversa = sum(dist_finale.query('gruppi_separati==True')['var_ratio'])
				peso_stessa_classe = sum(dist_finale.query('gruppi_separati==False')['var_ratio'])
				
				
				
				
								
				numerosità.append(a)
				rumore.append(b)
				esponente.append(p)
				
				sovrapposizione_classe_diversa.append(peso_classe_diversa)
				sovrapposizione_stessa_classe.append(peso_stessa_classe)
				
				perc = peso_classe_diversa/(peso_classe_diversa + peso_stessa_classe)
				percentuale.append(perc)
				
		
				#aa = np.array(sovrapposizione_classe_diversa)
				#bb = np.array(sovrapposizione_stessa_classe)
				#perc = aa/(aa+bb)
				
				
#%%

df_confronto_varianza = {'N':numerosità,'sigma':rumore,'esponente':esponente,'perc':perc} 
df_confronto_varianza = pd.DataFrame(df_confronto_varianza)

			

#%%
import pylab as plt

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

ax.plot(n,perc)
ax.loglog()





