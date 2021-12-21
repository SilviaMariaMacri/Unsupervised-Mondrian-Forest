# cd "C:\Users\silvi\Desktop\Fisica\TESI\Tesi"

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,cdist
#import pylab as plt
import seaborn as sns




df = pd.read_csv('confronto_varianza.txt',sep='\t')

#%%

#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
#ax.plot(df['sigma'],df['perc'])
#ax.loglog()


g=sns.relplot(x="sigma", 
			y="perc", 
			data=df, 
			kind="line",
			col='N',
			hue='esponente',
			)
#g.set(xscale="log",yscale="log")



#%%


fig,ax = plt.subplots()

ax.scatter(x1[0],x1[1])
ax.scatter(x2[0],x2[1])



#%%



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
n = np.geomspace(0.01, 1, 11) # rumore
pot = [1,5,10,20,30,40,50] #esponente rapporto varianze

numerosità = []
rumore = []
esponente = []
percentuale = []

sovrapposizione_classe_diversa = []
sovrapposizione_stessa_classe = []





for a in N:
		
	
	for b in n:
		
 		
		for p in  pot:
			print('N: ',a)
			print('rumore: ',b)
			print('esponente: ',p)
			
			for rip in range(6):
				
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

				


df_confronto_varianza = {'N':numerosità,'sigma':rumore,'esponente':esponente,'perc':percentuale} 
df_confronto_varianza = pd.DataFrame(df_confronto_varianza)

