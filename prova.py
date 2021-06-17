

def MondrianPolygon_SingleCut(t0,l,lifetime,father):
	

	# array di lunghezze intervalli
	ld = []
	for i in l:
		ld.append(distance.euclidean(i[0],i[1]))

		
	# linear dimension
	LD = sum(ld)  

	n_vert = np.arange(len(l))



	# genera tempo per cut
	time_cut = np.random.exponential(1/LD)


	t0 += time_cut
	
	if t0 > lifetime:
		return
		
		
	p=ld/sum(np.array(ld))
	lati = choice(n_vert,p=p,replace=False,size=2)
	lati.sort()

	l1 = l.copy()
	l2 = l.copy()
		
		
	point1 = np.array(l[lati[0]][0]) + (np.array(l[lati[0]][1]) - np.array(l[lati[0]][0]))*np.random.uniform(0,1)
	point2 = np.array(l[lati[1]][0]) + (np.array(l[lati[1]][1]) - np.array(l[lati[1]][0]))*np.random.uniform(0,1)

			
	
	
		
	
		
	l1 = []
	l1.append([point1,point2])
	for i in range(lati[1],len(l)):
		if i == lati[1]:
			l1.append([point2,l[lati[1]][1]])
		else:
			l1.append(l[i])
				
	for i in range(lati[0]+1):
		if i == lati[0]:
			l1.append([l[lati[0]][0],point1])
		else:
			l1.append(l[i])
			
	l2 = []
	l2.append([point2,point1])
	for i in range(lati[0],lati[1]+1):
		if i == lati[0]:
			l2.append([point1,l[lati[0]][1]])
		if i == lati[1]:
			l2.append([l[lati[1]][0],point2])
		if (i != lati[0]) & (i != lati[1]):
			l2.append(l[i])
	
	
	
	risultato1 = [t0, l1]
	risultato2 = [t0, l2]
	risultato = [risultato1, risultato2, t0, father]
	
	
	
	return risultato










def MondrianPolygon(t0,vertici_iniziali,lifetime): 
	
	
	

	m=[]
	count_part_number = 0
	m0 = [ t0,vertici_iniziali,count_part_number ] 
	m.append(m0)
	
	box = []
	time = []
		
	father = []
	part_number = []
	
	#vertici = []
	#vertici.append(vertici_iniziali)
	
	
	
	
	vertici_per_plot=[]
	for i in range(len(vertici_iniziali)):
		vertici_per_plot.append(vertici_iniziali[i][0])
	box.append(vertici_per_plot)
	time.append(t0)
	father.append('nan')
	part_number.append(count_part_number)
	
			
			
		
	
	
	for i in m:

	
		try:
			

			 
		 
			mondrian = MondrianPolygon_SingleCut(i[0],i[1],lifetime,i[2])
			
			m.append([mondrian[0][0],mondrian[0][1],count_part_number+1])
			count_part_number += 1
			part_number.append(count_part_number)
			
			m.append([mondrian[1][0],mondrian[1][1],count_part_number+1])
			count_part_number += 1
			part_number.append(count_part_number)
			
			
	
		
			for j in range(2):
				vertici_per_plot=[]
				for i in range(len(mondrian[j][1])):
					vertici_per_plot.append(mondrian[j][1][i][0])
				#vertici.append(mondrian[j][1])
				box.append(vertici_per_plot)
				time.append(mondrian[2])
				father.append(mondrian[3])
				



		except  TypeError:
			continue
		
	df = {'time':time,'father':father,'part_number':part_number}#,'vertici':vertici}
	df = pd.DataFrame(df)
	
	

	leaf = []
	for i in range(len(df)):
		if df['part_number'].iloc[i] not in df['father'].unique():
			leaf.append(True)
		else:
			leaf.append(False)
		
			
		
	df['leaf'] = leaf

		
	
	
	return m,box,df
		






#m = (y1-y2) / (x1-x2)
#q = (x1*y2 - x2*y1) / (x1-x2)






#%%

import numpy as np
import pandas as pd	
import random




def Mondrian(t0,d,l,lifetime):
	
	
	# array di lunghezze intervalli
	ld = []
	for i in l:
		ld.append(i[1]-i[0])
		
		
	# linear dimension
	LD = sum(ld)  
	




	# genera tempo per cut
	time_cut = np.random.exponential(1/LD)
	t0 += time_cut
	
	if t0 < lifetime:
		
		
		# genera dimesione cut
		d_cut = random.choices(d, weights=ld, k=1)


		# cut
		x = np.random.uniform(l[d_cut[0]][0],l[d_cut[0]][1])


		l_min = l.copy()
		l_max = l.copy()

		l_min[d_cut[0]] = [l[d_cut[0]][0],x]
		l_max[d_cut[0]] = [x,l[d_cut[0]][1]] 


		risultato1 = [t0, l_min]
		risultato2 = [t0, l_max]
		risultato = [risultato1, risultato2, x,t0,d_cut]
		return risultato


	else:
		return






def Mondrian_completo(t0,numero_dimensioni,intervallo,lifetime): #tutti intervalli uguali

	
	# dimensioni
	d = np.arange(numero_dimensioni)
	 
	# array di intervalli  
	l=[]
	for i in d:
		l.append(intervallo)
		
	
	mondrian = Mondrian(t0,d,l,lifetime)
	m=[]
	m.append(mondrian[0])
	m.append(mondrian[1])
	
	
	#solo se d=[0,1]
	d1x = []
	d1x.append(mondrian[0][1][0][0])
	d1y = []
	d1y.append(mondrian[0][1][0][1])
	d2x = []
	d2x.append(mondrian[0][1][1][0])
	d2y = []
	d2y.append(mondrian[0][1][1][1])
	
	
	tempo=[]
	tempo.append(mondrian[3])
	
	cut=[]
	cut.append(mondrian[2])
	
	dim = []
	dim.append(mondrian[4][0])	
	

	
	
	for i in m:
	
		try:
			mondrian = Mondrian(i[0],d,i[1],lifetime)
			m.append(mondrian[0])
			m.append(mondrian[1])
			
			#solo se d=[0,1]
			d1x.append(mondrian[0][1][0][0])
			d1y.append(mondrian[0][1][0][1])
			d2x.append(mondrian[0][1][1][0])
			d2y.append(mondrian[0][1][1][1])
			
			
			tempo.append(mondrian[3])
			cut.append(mondrian[2])
			dim.append(mondrian[4][0])


			
		except  TypeError:
				continue
			
			
			
	return m,tempo,cut,dim,d1x,d1y,d2x,d2y
			






			
#%%
		


# tempo iniziale
t0 = 0
numero_dimensioni = 2
intervallo = [0,1]  # considero intervalli tutti uguali
lifetime=2.5

m,tempo,cut,dim,d1x,d1y,d2x,d2y = Mondrian_completo(t0,numero_dimensioni,intervallo,lifetime)


#%%

t0=0
lifetime=2

df1 = pd.DataFrame(iris.data)
X=np.array(df1[[0,1]])

m,tempo,cut,dim,d1x,d1y,d2x,d2y = Mondrian_completo(t0,X,spazio_iniziale,lifetime)


#%%
df = {'tempo':tempo,'cut':cut,'dim':dim,'d0x':d1x,'d0y':d1y,'d1x':d2x,'d1y':d2y}
df = pd.DataFrame(df)



#%%    due dimensuiioni 


import matplotlib.pylab as plt




fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

for i in range(len(df[df['dim']==0])):
	ax.vlines(df.query('dim==0')['cut'].iloc[i],df.query('dim==0')['d1x'].iloc[i],df.query('dim==0')['d1y'].iloc[i],color='b')#, linestyle = '--', color=color[i], linewidth=1) # - QD_peaks[i][1]
	#ax.text(df.query('dim==0')['cut'].iloc[i],df.query('dim==0')['d1x'].iloc[i],df.query('dim==0')['tempo'].iloc[i].round(4),color='b')#,  fontsize=12, color=color[i])


for i in range(len(df[df['dim']==1])):
	ax.hlines(df.query('dim==1')['cut'].iloc[i],df.query('dim==1')['d0x'].iloc[i],df.query('dim==1')['d0y'].iloc[i],color='orange')#, linestyle = '--', color=color[i], linewidth=1) # - QD_peaks[i][1]
	#ax.text(df.query('dim==1')['d0x'].iloc[i],df.query('dim==1')['cut'].iloc[i],df.query('dim==1')['tempo'].iloc[i].round(4))#,  fontsize=12, color=color[i])


#ax.scatter(df1[0],df1[1])

#ax.set_xlim(0,1)
#ax.set_ylim(0,1)





#%%







# Mondrian(i[0],df,d,i[1],lifetime)
def Mondrian(t0,df,d,l,lifetime):


	# array di lunghezze intervalli 
	ld = []
	for i in l:
		ld.append(i[1]-i[0])
		
		
	# linear dimension
	LD = sum(ld)  
	print(LD)




	# genera tempo per cut
	time_cut = np.random.exponential(1/LD)
	t0 += time_cut
	

	if t0 < lifetime:


		# genera dimesione cut
		d_cut = random.choices(d, weights=ld, k=1)

		if l[d_cut[0]][0] != l[d_cut[0]][1]:
			
			# cut
			x = np.random.uniform(l[d_cut[0]][0],l[d_cut[0]][1])


			l_min = l.copy()
			l_max = l.copy()
		

		
			df = df[d_cut] 
		
		
			l_min[d_cut[0]] = [l[d_cut[0]][0],df[df[d_cut]<x].max().iloc[0]]
			risultato1 = [t0, l_min]
			
			l_max[d_cut[0]] = [df[df[d_cut]>x].min().iloc[0],l[d_cut[0]][1]]
			risultato2 = [t0, l_max]
			
			
			risultato = [risultato1, risultato2, x, t0, d_cut]
		
			print(l_max[d_cut[0]])
			print(l_min[d_cut[0]])
		
			return risultato


	else:
		return









def Mondrian_completo(t0,X,lifetime): 
	
	


	df = pd.DataFrame(X)
	d = np.arange(len(df.columns))	



	l = []
	for i in df.columns:
		l.append([df[i].min(),df[i].max()])



	

	mondrian = Mondrian(t0,df,d,l,lifetime)
	
	m=[]
	m.append(mondrian[0])
	m.append(mondrian[1])
	
	
	#solo se d=[0,1]
	d1x = []
	d1x.append(mondrian[0][1][0][0])
	d1y = []
	d1y.append(mondrian[0][1][0][1])
	d2x = []
	d2x.append(mondrian[0][1][1][0])
	d2y = []
	d2y.append(mondrian[0][1][1][1])
	
	
	tempo=[]
	tempo.append(mondrian[3])
	
	cut=[]
	cut.append(mondrian[2])
	
	dim = []
	dim.append(mondrian[4][0])	
	



	for i in m:
	
		try:
			
			
			mondrian = Mondrian(i[0],df,d,i[1],lifetime)
			m.append(mondrian[0])
			m.append(mondrian[1])
			
			#solo se d=[0,1]
			d1x.append(mondrian[0][1][0][0])
			d1y.append(mondrian[0][1][0][1])
			d2x.append(mondrian[0][1][1][0])
			d2y.append(mondrian[0][1][1][1])
			
			
			tempo.append(mondrian[3])
			cut.append(mondrian[2])
			dim.append(mondrian[4][0])



		except  TypeError:
			continue




			
	return m,tempo,cut,dim,d1x,d1y,d2x,d2y
			

	

#%%



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
	
	








