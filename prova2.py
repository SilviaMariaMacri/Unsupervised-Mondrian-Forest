problemi:
	i tagli hanno senso come vengono? forse probabilità scelta dimensione è sbagliata




#%%
import numpy as np
import random
import pandas as pd

#%% 

# l = estremi degli intervalli
# data = dataframe di dati

def Mondrian(data,t0,l,lifetime):


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
	

	if t0 < lifetime:
 

		# genera dimesione cut
		d_cut = random.choices(d, weights=ld, k=1)[0] 
		
		

		if data[d_cut].max() != data[d_cut].min():
			
			# cut
			x = np.random.uniform(data[d_cut].max(),data[d_cut].min())


			l_min = l.copy()
			l_max = l.copy()
			l_min[d_cut] = [l[d_cut][0],x]
			l_max[d_cut] = [x,l[d_cut][1]]

			risultato1 = [t0, l_min]
			risultato2 = [t0, l_max]
			
			
			risultato = [risultato1, risultato2, x, t0, d_cut]




		
			return risultato


	else:
		return









def Mondrian_completo(dat,t0,spazio_iniziale,lifetime): 
	
	
	data = dat[np.arange(len(spazio_iniziale))]
	

	m=[]
	m0 = [ t0,spazio_iniziale ]
	m.append(m0)


	intervalli = []
	x = []
	tempo = []
	dim = []
	
	for i in range(len(spazio_iniziale)):
		for j in range(2):
			
			intervalli.append(np.reshape(spazio_iniziale,(1,len(spazio_iniziale)*2))[0])
			x.append(spazio_iniziale[i][j])
			tempo.append(t0)
			dim.append(i)
		
		
		
	for i in m:
	
		try:
			
			
			mondrian = Mondrian(data,i[0],i[1],lifetime)
			m.append(mondrian[0])
			m.append(mondrian[1])
			
		
			
			intervalli.append(np.reshape(mondrian[0][1],(1,len(spazio_iniziale)*2))[0])
			x.append(mondrian[2])
			tempo.append(mondrian[3])
			dim.append(mondrian[4])



		except  TypeError:
			continue
		
		
	
	df_intervalli = pd.DataFrame(intervalli)
	nomi = []
	for i in range(len(spazio_iniziale)):
		for j in ['min','max']:
			nomi.append(str(i)+j)
	df_intervalli.columns = nomi

	
	df_altro = {'x':x,'tempo':tempo,'dim':dim}
	df_altro = pd.DataFrame(df_altro)
	
	valori_per_partizione = pd.merge(df_altro, df_intervalli, left_index=True, right_index=True)
		
		
		
	




			
	return m,valori_per_partizione
			











#%%





from itertools import product





def Partizione(df):
	


	
	x_per_dim = []
	for i in df['dim'].unique():
		nomi_colonne=[]
		for j in df.columns:
			if (j.startswith(str(i))==False):
				nomi_colonne.append(j)
	
		x_con_limiti = df[df['dim']==i].sort_values('x')[nomi_colonne]
		x_con_limiti.index = np.arange(0,len(x_con_limiti),1)
		x_con_limiti.columns = [str(x_con_limiti['dim'].unique()[0])+'x' if k=='x' else k for k in x_con_limiti.columns]
		
		x_per_dim.append(x_con_limiti)
	




	
	
	




	n_d = len(x_per_dim)
	
	
	
	point_tot = pd.DataFrame()
	for i in range(n_d):  # scelgo una dimensione per il cut
		point_df = pd.DataFrame()
		
		for j in range(len(x_per_dim[i])):   # per ogni riga del dataframe
			point=[]
			
			dim = list(np.arange(n_d))
			dim_r = dim.copy()
			dim_r.remove(i)
			
			for lst in product(*([x_per_dim[i][str(n)+'min'].iloc[j],x_per_dim[i][str(n)+'max'].iloc[j]] for n in dim_r)):
				   point.append(list(lst))
			
			point = pd.DataFrame(point)
			point.columns = dim_r
			point[i] = x_per_dim[i][str(i)+'x'].iloc[j]
			point = point[dim]	
			point_df = pd.concat([point_df,point])#.sort_values(by=dim)
			
		point_tot =  pd.concat([point_df,point_tot]).sort_values(by=dim)
			
	
		point_tot = point_tot.drop_duplicates()
		point_tot.index = np.arange(0,len(point_tot),1)

		
	




	w = point_tot.copy()

	lim_tot = pd.DataFrame()


	for i in range(len(w)):
	
	
	
		min_limit=[]
		#w_with_fixed_min = []
		#w_without_fixed_min = pd.DataFrame()
		
		for j in range(n_d):
			
			lim_min = w[j].iloc[i]
			min_limit.append(lim_min) #tutti i limiti  minimi
			
			#w_with_fixed_min.append(w[w[j]==lim_min])
			#w_without_fixed_min = pd.concat([w[w[j]!=lim_min],w_without_fixed_min])
			
			
		
				
		
		# tutte dimensioni fissate tranne una
		vary_one_dimension = [] 
		for l in range(n_d):
			b=[]
			for j in range(len(w)):
				if w[l].iloc[i] != max(w[l]):
					if w[l].iloc[j]>w[l].iloc[i]:
	
						
						dim = list(np.arange(n_d))
						dim_r = dim.copy()
						dim_r.remove(l)
	
						prova=[]
						for k in dim_r:
							if w[k].iloc[j]==w[k].iloc[i]:
								prova.append(0)
						if len(prova) == (n_d-1):
							b.append(w[l].iloc[j])
			vary_one_dimension.append(b)
						
		
		possible_max = []
		for lst in product(*(vary_one_dimension[n] for n in range(n_d))):
			possible_max.append(list(lst))
			
		possible_max_real = []
		for j in range(len(possible_max)):
			for k in range(len(w)):
				if list(w.iloc[k]) == possible_max[j]:
					possible_max_real.append(possible_max[j])
			
		
		sum_max = []
		for j in range(len(possible_max_real)):
			sum_max.append(sum(possible_max_real[j]))



		try:
			min_sum_max = min(sum_max)
			
		except ValueError:
			continue
		
		index_max = sum_max.index(min_sum_max)
		
		
		
		max_limit = possible_max_real[index_max]
		
		

		
		name_columns_min = []
		for j in range(n_d):
			name_columns_min.append(str(j)+'min')

		name_columns_max = []
		for j in range(n_d):
			name_columns_max.append(str(j)+'max')
			
		name_columns = np.concatenate([name_columns_min,name_columns_max])
		
		
		
		
		
		lim = np.concatenate([min_limit,max_limit])
		lim = pd.DataFrame(lim).T
		lim.columns = name_columns
		
		
		
		lim_tot = pd.concat([lim_tot,lim])
		
				




	lim_tot.index=np.arange(len(lim_tot))
	

	
	return lim_tot,w
	
			
					


#%%




def Count(lim,data):	
	
	count_class=[] 
	
	for l in data['class'].unique():
		
		dat = data[data['class']==l]
		dat.index=np.arange(len(dat))
		
	
		data_count = []	
		n_d = int(len(lim.columns)/2)
		
		for k in range(len(lim)):
			count = 0
			for i in range(len(dat)):
				partial_count=[]
				for j in range(n_d):
					if (dat.iloc[i][j]>lim[str(j)+'min'].iloc[k]) & (dat.iloc[i][j]<lim[str(j)+'max'].iloc[k]):
						partial_count.append(0)
					else:
						break
				if len(partial_count) == n_d:
					count += 1
			data_count.append(count)	
		
		count_class.append(data_count)
		
		
	for i in range(len(count_class)):
		lim[str(i)+'counts'] = count_class[i]
		
#devi aggiungee colonna classificazione associata a partizione		
		
	return lim


#%%






def Class(lim_class,X):	
	
	X = pd.DataFrame(X)
	
	cl = []
	
			#lim_class[str(j)+'data'] = i[j]
			#lim_class.eval("find_class = ("+str(j)+"data>"+str(j)+"min) and ("+str(j)+"data<"+str(j)+"max)", inplace=True)
	
	
	for i in X:
		for j in range(len(lim_class)):
			for k in range(len(X[0])):
				if (X[k]>lim_class[str(k)+'min'].iloc[j]) & (X[k]<lim_class[str(j)+'max'].iloc[j]):
					partial_count.append(0)
				else:
					break
			if len(partial_count) == len(X[0]):
				cl.append(lim_class['class'].iloc[j])
				break
			
				
	X['class_data'] = cl
		
	return cl




	


#%% 













	
	







#%%
from sklearn import datasets
 
dat = datasets.make_moons(n_samples=100,noise=0.2)
iris = datasets.load_iris()



#make_moons
data = pd.DataFrame(dat[0])
data['class']=dat[1]
X = dat[0]

#iris
#data = pd.DataFrame(iris.data)
#data[[0,1]]#,2]]


t0=0
spazio_iniziale = [ [data[0].min(),data[0].max()],[data[1].min(),data[1].max()] ]     
                  #[[4, 8], [2, 5]]#, [1,7]]  
lifetime=2



#%%

#crea partizione a partire dai dati
m,df = Mondrian_completo(data,t0,spazio_iniziale,lifetime)
lim,w = Partizione(df)
#per ogni classe, conta i punti all'interno di ogni partizione
lim_class = Count(lim,data)
#classifica ogni dato non precedentemente classificato a seconda della partizione




#%%


import matplotlib.pylab as plt




fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

for i in range(len(df[df['dim']==0])):
	ax.vlines(df.query('dim==0')['x'].iloc[i],df.query('dim==0')['1min'].iloc[i],df.query('dim==0')['1max'].iloc[i],color='b')#, linestyle = '--', color=color[i], linewidth=1) # - QD_peaks[i][1]
	#ax.text(df.query('dim==0')['cut'].iloc[i],df.query('dim==0')['d1x'].iloc[i],df.query('dim==0')['tempo'].iloc[i].round(4),color='b')#,  fontsize=12, color=color[i])


for i in range(len(df[df['dim']==1])):
	ax.hlines(df.query('dim==1')['x'].iloc[i],df.query('dim==1')['0min'].iloc[i],df.query('dim==1')['0max'].iloc[i],color='orange')#, linestyle = '--', color=color[i], linewidth=1) # - QD_peaks[i][1]
	#ax.text(df.query('dim==1')['d0x'].iloc[i],df.query('dim==1')['cut'].iloc[i],df.query('dim==1')['tempo'].iloc[i].round(4))#,  fontsize=12, color=color[i])


#ax.scatter(data[0],data[1])

ax.scatter(data[data['class']==0][0],data[data['class']==0][1])
ax.scatter(data[data['class']==1][0],data[data['class']==1][1])




#%%
	
	
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

for i in lim.index:
	
	ax.vlines(lim['0min'].iloc[i],lim['1min'].iloc[i],lim['1max'].iloc[i])		
	ax.vlines(lim['0max'].iloc[i],lim['1min'].iloc[i],lim['1max'].iloc[i])
	ax.hlines(lim['1min'].iloc[i],lim['0min'].iloc[i],lim['0max'].iloc[i])
	ax.hlines(lim['1max'].iloc[i],lim['0min'].iloc[i],lim['0max'].iloc[i])
	
	
	
ax.scatter(w[0],w[1],color='r')

#ax.scatter(data[data['class']==0][0],data[data['class']==0][1])
#ax.scatter(data[data['class']==1][0],data[data['class']==1][1])






















	
		



			
			
			
			
			
			
			
			
			
		
		
	
		
		
		
			














#%%





import mondrianforest
from sklearn import datasets, model_selection


iris = datasets.load_iris()
forest = mondrianforest.MondrianTreeClassifier()

cv = model_selection.ShuffleSplit(n_splits=20, test_size=0.10)
scores = model_selection.cross_val_score(forest, iris.data, iris.target, cv=cv)

print(scores.mean(), scores.std())
































#%%     calcolare partizione


#########  solo per  dim = 2



def Partizione(df):
	


	
	x_per_dim = []
	for i in df['dim'].unique():
		nomi_colonne=[]
		for j in df.columns:
			if (j.startswith(str(i))==False):
				nomi_colonne.append(j)
	
		x_con_limiti = df[df['dim']==i].sort_values('x')[nomi_colonne]
		x_con_limiti.index = np.arange(0,len(x_con_limiti),1)
		x_con_limiti.columns = [str(x_con_limiti['dim'].unique()[0])+'x' if k=='x' else k for k in x_con_limiti.columns]
		
		x_per_dim.append(x_con_limiti)
	




	# da qui è solo per due dimensioni


	x=x_per_dim[0]
	y=x_per_dim[1]


	hx=[]
	for i in range(len(x)):
		a = [x['0x'].iloc[i],x['1min'].iloc[i]]
		b = [x['0x'].iloc[i],x['1max'].iloc[i]]
		hx.append(a)
		hx.append(b)
	
	hx = pd.DataFrame(hx)	
	
	
	hy=[]
	for i in range(len(y)):
		a = [y['0min'].iloc[i],y['1x'].iloc[i]]
		b = [y['0max'].iloc[i],y['1x'].iloc[i]]
		hy.append(a)
		hy.append(b)
	
	hy = pd.DataFrame(hy)		





	# dataframe con estremi partizioni ordinati
	w = pd.concat([hx,hy]).sort_values(by=[0,1])
	w.index=np.arange(0,len(w),1)





	a=[]
	for i in range(len(w)):
		for j in range(len(w)):
			if j>i:
				if (w[0].iloc[i]==w[0].iloc[j]) & (w[1].iloc[i]==w[1].iloc[j]):
					a.append(j)
			
		
	w=w.drop(a)
			
	w.index=np.arange(0,len(w),1)





	estremi=[]


	for i in range(len(w)):
	
	
	#if w[0].iloc[i] != max(w[0]):
	
	
	
		a=[]
		a.append(w[0].iloc[i]) #x_min
		b=[]
		for j in range(len(w)):
			if w[0].iloc[i] != max(w[0]):
				if (w[0].iloc[j]>w[0].iloc[i]) & (w[1].iloc[j]==w[1].iloc[i]):
					
					
					b.append(w[0].iloc[j])
					

		#a.append(w[0].iloc[j])
					#break
					
					 
		a.append(w[1].iloc[i]) #y_min 
		
		
	
		w1 = w[w[0]==a[0]]
		
		w2=[]
		for j in range(len(b)):
			w2.append(w[w[0]==b[j]])
			
		
		for j in range(len(w2)):
			w12 = pd.merge(w1.set_index(1),w2[j].set_index(1), left_index=True, right_index=True)
			w12 = w12[w12.index>w[1].iloc[i]]
			
			if w12.empty !=True:
				
				a.append(b[j]) #x_max
				a.append(min(w12[w12.index>w[1].iloc[i]].index)) #y_max
			
				estremi.append(a)
				break





	estremi_df = pd.DataFrame(estremi)#.dropna()
	estremi_df.columns = ['x_min','y_min','x_max','y_max']

	estremi_df.index=np.arange(0,len(estremi_df),1)
	
	
	
	return estremi_df,w
	
	
	
	




