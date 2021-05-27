import numpy as np
import random
import pandas as pd
import matplotlib.pylab as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#%%

 
dat = datasets.make_moons(n_samples=300,noise=0.1)
iris = datasets.load_iris()



#make_moons
data = pd.DataFrame(dat[0])


# 3D
altra_dim = np.random.normal(0, 1, len(data))
data[2] = altra_dim


data['class']=dat[1]
X = dat[0]

'''
#iris
data = pd.DataFrame(iris.data)
data[[0,1,2]]

'''

t0=0

# 2D
#spazio_iniziale = [ [data[0].min(),data[0].max()],[data[1].min(),data[1].max()] ]     
# 3D
spazio_iniziale = [ [data[0].min(),data[0].max()],[data[1].min(),data[1].max()],[data[2].min(),data[2].max()] ]     

lifetime=3


#%%


X = np.array(data[[0,1,2]]).reshape((len(data),3))
y = np.array(data['class'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)








#%%



#crea partizione a partire dai dati
part = Mondrian_completo(X_train,y_train,t0,spazio_iniziale,lifetime)
part_with_counts = Count(X_train,y_train,part)
cl = Class(X_test,part_with_counts)
#PartitionPlot3D(X_train,y_train,part)
PartitionPlot3D(X_train,y_train,part)






#%% 




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
	
	
	if t0 > lifetime:
		return
	
	
	if len(data['class'].unique())<2:
		return
	
	
	# genera dimesione cut
	d_cut = random.choices(d, weights=ld, k=1)[0] 
		
	
	if data[d_cut].max() == data[d_cut].min():
		return
	
	
	
	# cut
	#x = np.random.uniform(data[d_cut].max(),data[d_cut].min())
	clf = LogisticRegression(penalty='none').fit(np.array(data[d_cut]).reshape((len(data),1)), np.array(data['class']).reshape((len(data),1)))
	x = -clf.intercept_[0]/clf.coef_[0][0]
	
	
	
	if (x<l[d_cut][0]) or (x>l[d_cut][1]):
		return
				
				
				
			
	l_min = l.copy()
	l_max = l.copy()
	l_min[d_cut] = [l[d_cut][0],x]
	l_max[d_cut] = [x,l[d_cut][1]]
			

	risultato1 = [t0, l_min]
	risultato2 = [t0, l_max]
			
			
		
			
			
	risultato = [risultato1, risultato2, x, t0, d_cut, father]




		
	return risultato
		
		










def Mondrian_completo(X,y,t0,spazio_iniziale,lifetime): 
	
	
	
	data = pd.DataFrame(X)
	data['class'] = y
	
	

	m=[]
	count_part_number = 0
	m0 = [ t0,spazio_iniziale,count_part_number ] 
	m.append(m0)
	
	
	box = []
	#x = []
	time = []
	#dim = []
		
	father = []
	part_number = []
	
	
	box.append(np.reshape(spazio_iniziale,(1,len(spazio_iniziale)*2))[0])
	#x.append('nan')
	#dim.append('nan')		
	time.append(t0)
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
			
			
	
		
			for j in range(2):
				box.append(np.reshape(mondrian[j][1],(1,len(spazio_iniziale)*2))[0])
				#x.append(mondrian[2])
				time.append(mondrian[3])
				#dim.append(mondrian[4])
				father.append(mondrian[5])
				



		except  TypeError:
			continue
		
		
	
	
	names = []
	for i in range(len(spazio_iniziale)):
		for j in ['min','max']:
			names.append(str(i)+j)
	
	
	df_box = pd.DataFrame(box)
	df_box.columns = names	
	df = {'time':time,'father':father,'part_number':part_number}
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

			




















def Count(X,y,part): 	
	
	
	data = pd.DataFrame(X)
	n_d = len(data.columns)
	data['class'] = y
	

	
	p = part[part['leaf']==True].copy()
	
	count_class=[] 
	
	for l in data['class'].unique():
		
		dat = data[data['class']==l]
		dat.index=np.arange(len(dat))


		data_count = []	
		
		
		for k in range(len(p)):
			count = 0
			for i in range(len(dat)):
				partial_count=[]
				for j in range(n_d):
					if (dat.iloc[i][j]>p[str(j)+'min'].iloc[k]) & (dat.iloc[i][j]<p[str(j)+'max'].iloc[k]):
						partial_count.append(0)
					else:
						break
				if len(partial_count) == n_d:
					count += 1
			data_count.append(count)	
		
		count_class.append(data_count)
		
		
	for i in range(len(count_class)):
		p[str(i)+'counts'] = count_class[i]
		
		
		

	
		
	return p









def Class(X,part_with_counts):	
	
	
	
	cl0 = []
	cl1 = []
	part_number = []
	
			#part_with_counts[str(j)+'data'] = i[j]
			#part_with_counts.eval("find_class = ("+str(j)+"data>"+str(j)+"min) and ("+str(j)+"data<"+str(j)+"max)", inplace=True)
	
	
	for i in X:
		count=0
		for j in range(len(part_with_counts)):
			count += 1
			partial_count=[]
			for k in range(len(X[0])):
				if (i[k]>part_with_counts[str(k)+'min'].iloc[j]) & (i[k]<part_with_counts[str(k)+'max'].iloc[j]):
					partial_count.append(0)
				else:
					break
			if len(partial_count) == len(X[0]):
				cl0.append(part_with_counts['0counts'].iloc[j])  
				cl1.append(part_with_counts['1counts'].iloc[j]) 
				part_number.append(part_with_counts['part_number'].iloc[j]) 
				break
			else:
				if count==len(part_with_counts):
					cl0.append('nan')
					cl1.append('nan')
					part_number.append('nan')
					
					

	X = pd.DataFrame(X)			
	X['0counts_data'] = cl0
	X['1counts_data'] = cl1
	X['part_number_data'] = part_number
	
	
	
		
	return X








from matplotlib.pyplot import cm




def PartitionPlot3D(X,y,part):
	
	
	data = pd.DataFrame(X)
	data['class'] = y

	
	p = part[part['leaf']==True]
	
	p = p[['0min','0max','1min','1max','2min','2max']]
	p.columns = ['xmin','xmax','ymin','ymax','zmin','zmax']
	
	p['indice'] = np.arange(1,len(p)+1,1)
	
	
	p = (p.eval("xmin_r = xmin + (xmax-xmin)*0.05")
		 .eval("ymin_r = ymin + (ymax-ymin)*0.05")
		 .eval("zmin_r = zmin + (zmax-zmin)*0.05")
		 .eval("xmax_r = xmax - (xmax-xmin)*0.05")
		 .eval("ymax_r = ymax - (ymax-ymin)*0.05")
		 .eval("zmax_r = zmax - (zmax-zmin)*0.05")
		 )


	
	p = p[['xmin_r','xmax_r','ymin_r','ymax_r','zmin_r','zmax_r']]
	p.columns = ['0min','0max','1min','1max','2min','2max']
	
	
	
	
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
				
				ax.plot([x_min,x_max],
			            [y_min,y_max],
						[z_min,z_max],
						color=c)
				
				
	ax.scatter(data[data['class']==0][0],data[data['class']==0][1],data[data['class']==0][2])
	ax.scatter(data[data['class']==1][0],data[data['class']==1][1],data[data['class']==1][2])
	
				
	plt.show()
	
	return

	
	
	
	
def PartitionPlot2D(X,y,part):
	
	
	data = pd.DataFrame(X)
	data['class'] = y


	
	p = part[part['leaf']==True]	
	
	
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
	
	for i in range(len(p)):
		
		ax.vlines(p['0min'].iloc[i],p['1min'].iloc[i],p['1max'].iloc[i])		
		ax.vlines(p['0max'].iloc[i],p['1min'].iloc[i],p['1max'].iloc[i])
		ax.hlines(p['1min'].iloc[i],p['0min'].iloc[i],p['0max'].iloc[i])
		ax.hlines(p['1max'].iloc[i],p['0min'].iloc[i],p['0max'].iloc[i])
		
		
		
	ax.scatter(data[data['class']==0][0],data[data['class']==0][1])
	ax.scatter(data[data['class']==1][0],data[data['class']==1][1])
	
	
	
	plt.show()
	
	return
	
	















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
	
	
 
clf = LogisticRegression(penalty='none').fit(np.array(X[0]).reshape((200,1)), np.array(X['cl']).reshape((200,1)))

clf.coef_
#array([[3.22627498]])

clf.intercept_
#array([-7.98655553])

