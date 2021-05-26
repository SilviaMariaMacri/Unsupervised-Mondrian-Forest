import numpy as np
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pylab as plt

 

# l = estremi degli intervalli
# data = dataframe di dati

def Mondrian(data,t0,l,lifetime,father):


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
			#clf = svm.SVC(kernel='linear') # Linear Kernel
			#clf.fit(data[d_cut], data['class'])
			x = np.random.uniform(data[d_cut].max(),data[d_cut].min())
			#x = 




			l_min = l.copy()
			l_max = l.copy()
			l_min[d_cut] = [l[d_cut][0],x]
			l_max[d_cut] = [x,l[d_cut][1]]
			

			risultato1 = [t0, l_min]
			risultato2 = [t0, l_max]
			
			
		
			
			
			risultato = [risultato1, risultato2, x, t0, d_cut, father]




		
			return risultato
		


	else:
		return









def Mondrian_completo(data,t0,spazio_iniziale,lifetime): 
	
	
	#data = dat[np.arange(len(spazio_iniziale))]
	
	

	m=[]
	count_part_number = 0
	m0 = [ t0,spazio_iniziale,count_part_number ] 
	m.append(m0)


	intervalli = []
	x = []
	tempo = []
	dim = []
	
	
	
	part_intervalli = []
	#part_x = []
	part_tempo = []
	#part_dim = []
		
	father = []
	part_number = []
	
	
	for i in range(len(spazio_iniziale)):
		for j in range(2):
			
			intervalli.append(np.reshape(spazio_iniziale,(1,len(spazio_iniziale)*2))[0])
			x.append(spazio_iniziale[i][j])
			tempo.append(t0)
			dim.append(i)
			
			
	#nuovo
	part_intervalli.append(np.reshape(spazio_iniziale,(1,len(spazio_iniziale)*2))[0])	
	part_tempo.append(t0)
	father.append('nan')
	part_number.append(count_part_number)
	
			
			
		
	
	
	for i in m:

	
		try:
			

			 
		 
			mondrian = Mondrian(data,i[0],i[1],lifetime,i[2])
			
			m.append([mondrian[0][0],mondrian[0][1],count_part_number+1])
			count_part_number += 1
			part_number.append(count_part_number)
			
			m.append([mondrian[1][0],mondrian[1][1],count_part_number+1])
			count_part_number += 1
			part_number.append(count_part_number)
			
			
			
			intervalli.append(np.reshape(mondrian[0][1],(1,len(spazio_iniziale)*2))[0])
			x.append(mondrian[2])
			tempo.append(mondrian[3])
			dim.append(mondrian[4])
			
			
			#nuovo			
			for j in range(2):
				part_intervalli.append(np.reshape(mondrian[j][1],(1,len(spazio_iniziale)*2))[0])
				#part_x.append(mondrian[2])
				part_tempo.append(mondrian[3])
				#part_dim.append(mondrian[4])
				father.append(mondrian[5])
				



		except  TypeError:
			continue
		
		
	
	
	nomi = []
	for i in range(len(spazio_iniziale)):
		for j in ['min','max']:
			nomi.append(str(i)+j)
	

	df_intervalli = pd.DataFrame(intervalli)
	df_intervalli.columns = nomi
	df_altro = {'x':x,'tempo':tempo,'dim':dim}
	df_altro = pd.DataFrame(df_altro)
	valori_per_partizione = pd.merge(df_altro, df_intervalli, left_index=True, right_index=True)
	
	
	df_part_intervalli = pd.DataFrame(part_intervalli)
	df_part_intervalli.columns = nomi	
	df_part = {'tempo':part_tempo,'father':father,'part_number':part_number}
	df_part = pd.DataFrame(df_part)
	#df_part.loc[ (df_part['part_number'] not in df_part['father']==True),'leaf'] = True
	#df_part.loc[*[(df_part['part_number'].iloc[i] not in df_part['father']) for i in range(len(df_part))]]	=True	


	leaf = []
	for i in range(len(df_part)):
		if df_part['part_number'].iloc[i] not in df_part['father'].unique():
			leaf.append(True)
		else:
			leaf.append(False)
		
			
		
	df_part['leaf'] = leaf
		
	
	partizione = pd.merge(df_part, df_part_intervalli, left_index=True, right_index=True)



	return partizione,valori_per_partizione
			

















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









def Class(lim_class,X):	
	
	
	
	cl = []
	
			#lim_class[str(j)+'data'] = i[j]
			#lim_class.eval("find_class = ("+str(j)+"data>"+str(j)+"min) and ("+str(j)+"data<"+str(j)+"max)", inplace=True)
	
	
	for i in X:
		count=0
		for j in range(len(lim_class)):
			count += 1
			partial_count=[]
			for k in range(len(X[0])):
				if (i[k]>lim_class[str(k)+'min'].iloc[j]) & (i[k]<lim_class[str(k)+'max'].iloc[j]):
					partial_count.append(0)
				else:
					break
			if len(partial_count) == len(X[0]):
				cl.append(lim_class['class'].iloc[j])  
				break
			else:
				if count==len(lim_class):
					cl.append('nan')
			
	X = pd.DataFrame(X)			
	X['class_data'] = cl
		
	return X





#%%
from matplotlib.pyplot import cm

def PartitionPlot(p,r):
	
	
	h1 = [['min','max'],
		  ['min','max'],
	      ['min','max'],
	      ['min','max']]
	h1 = pd.DataFrame(h1)
	
	
	
	h2 = [['min','min'],
	      ['min','min'],
	      ['max','max'],
	      ['max','max']]
	h2 = pd.DataFrame(h2)
	h2.columns = [2,3]
	
	
	
	h3 = [['min','min'],
	      ['max','max'],
	      ['min','min'],
	      ['max','max']]
	h3 = pd.DataFrame(h3)
	h3.columns = [4,5]
	
	
	h = pd.concat([h1,h2,h3],axis=1)
	
	order = [[0, 1, 2, 3, 4, 5],[2, 3, 0, 1, 4, 5],[2, 3, 4, 5, 0, 1]]
	
	
	
	p = partizione.copy()
	
	color=cm.rainbow(np.linspace(0,1,len(p)))
	ax = plt.axes(projection='3d')
	
	
	for i,c in zip(range(len(p)),color):
		
		for j in range(len(h)):
			for k in order:
				x_min = p['0'+h[k[0]].iloc[j]].iloc[i]
				x_max = p['0'+h[k[1]].iloc[j]].iloc[i]
				y_min = p['1'+h[k[2]].iloc[j]].iloc[i]
				y_max = p['1'+h[k[3]].iloc[j]].iloc[i]
				z_min = p['2'+h[k[4]].iloc[j]].iloc[i]
				z_max = p['2'+h[k[5]].iloc[j]].iloc[i]
				rx = r*(x_max-x_min)
				ry = r*(y_max-y_min)
				rz = r*(z_max-z_min)
				
				ax.plot([x_min+rx,x_max-rx],
			            [y_min+ry,y_max-ry],
						[z_min+rz,z_max-rz],
						color='b')
				
	plt.show()
	
	return


			
#%%
ax = plt.axes(projection='3d')
ax.plot([1,1],[1,2],[3,3])

	

						
#%%	
	


	ax = plt.axes(projection='3d')
	i=0
#ax.scatter3D(data[0], data[1], data[2])#, c=zdata, cmap='Greens');

#for i in range(len(lim)):
	ax.plot([lim['0min'].iloc[i],lim['0max'].iloc[i]],[lim['1min'].iloc[i],lim['1min'].iloc[i]],[lim['2min'].iloc[i],lim['2min'].iloc[i]])
	ax.plot([lim['0min'].iloc[i],lim['0max'].iloc[i]],[lim['1min'].iloc[i],lim['1min'].iloc[i]],[lim['2max'].iloc[i],lim['2max'].iloc[i]])
	ax.plot([lim['0min'].iloc[i],lim['0max'].iloc[i]],[lim['1max'].iloc[i],lim['1max'].iloc[i]],[lim['2min'].iloc[i],lim['2min'].iloc[i]])
	ax.plot([lim['0min'].iloc[i],lim['0max'].iloc[i]],[lim['1max'].iloc[i],lim['1max'].iloc[i]],[lim['2max'].iloc[i],lim['2max'].iloc[i]])

#for i in range(len(lim)):
	ax.plot([lim['0min'].iloc[i],lim['0min'].iloc[i]],[lim['1min'].iloc[i],lim['1max'].iloc[i]],[lim['2min'].iloc[i],lim['2min'].iloc[i]])
	ax.plot([lim['0min'].iloc[i],lim['0min'].iloc[i]],[lim['1min'].iloc[i],lim['1max'].iloc[i]],[lim['2max'].iloc[i],lim['2max'].iloc[i]])
	ax.plot([lim['0max'].iloc[i],lim['0max'].iloc[i]],[lim['1min'].iloc[i],lim['1max'].iloc[i]],[lim['2min'].iloc[i],lim['2min'].iloc[i]])
	ax.plot([lim['0max'].iloc[i],lim['0max'].iloc[i]],[lim['1min'].iloc[i],lim['1max'].iloc[i]],[lim['2max'].iloc[i],lim['2max'].iloc[i]])

#for i in range(len(lim)):
	ax.plot([lim['0min'].iloc[i],lim['0min'].iloc[i]],[lim['1min'].iloc[i],lim['1min'].iloc[i]],[lim['2min'].iloc[i],lim['2max'].iloc[i]])
	ax.plot([lim['0min'].iloc[i],lim['0min'].iloc[i]],[lim['1max'].iloc[i],lim['1max'].iloc[i]],[lim['2min'].iloc[i],lim['2max'].iloc[i]])
	ax.plot([lim['0max'].iloc[i],lim['0max'].iloc[i]],[lim['1min'].iloc[i],lim['1min'].iloc[i]],[lim['2min'].iloc[i],lim['2max'].iloc[i]])
	ax.plot([lim['0max'].iloc[i],lim['0max'].iloc[i]],[lim['1max'].iloc[i],lim['1max'].iloc[i]],[lim['2min'].iloc[i],lim['2max'].iloc[i]])


#%%   prova logistic ecc ecc
mean1 = (1, 1)
cov1 = [[1, 0], [0, 1]]
x1 = np.random.multivariate_normal(mean1, cov1, 100)
x1=pd.DataFrame(x1)
x1['cl']=0


mean2 = (4, 4)
cov2 = [[1, 0], [0, 1]]
x2 = np.random.multivariate_normal(mean2, cov2, 100)
x2=pd.DataFrame(x2)
x2['cl']=1


X = pd.concat([x1,x2])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
#ax.scatter(X[0],X[1])
ax.scatter(x1[0],x1[1])
	
ax.scatter(x2[0],x2[1])
	
	
#%%

from sklearn.linear_model import LogisticRegression

 
clf = LogisticRegression(penalty='none').fit(np.array(X[0]).reshape((200,1)), np.array(X['cl']).reshape((200,1)))

clf.coef_
#array([[3.22627498]])

clf.intercept_
#array([-7.98655553])





x= -intercept/coef



	
	







#%%
from sklearn import datasets
 
dat = datasets.make_moons(n_samples=100,noise=0.2)
iris = datasets.load_iris()


'''
#make_moons
data = pd.DataFrame(dat[0])
data['class']=dat[1]
X = dat[0]
'''
#iris
data = pd.DataFrame(iris.data)
data[[0,1,2]]



t0=0
#spazio_iniziale = [ [data[0].min(),data[0].max()],[data[1].min(),data[1].max()] ]     
spazio_iniziale = [[4, 8], [2, 5], [1,7]]  
lifetime=0.4

#%%


#crea partizione a partire dai dati
partizione,df = Mondrian_completo(data,t0,spazio_iniziale,lifetime)
lim,w = Partizione(df)
#lim=lim[['0min','0max','1min','1max','2min','2max']]
#per ogni classe, conta i punti all'interno di ogni partizione
#lim_class = Count(lim,data)
#classifica ogni dato non precedentemente classificato a seconda della partizione
# Class(lim_class,X)+





#%%







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
	
	
	
	




