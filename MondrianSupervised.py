import numpy as np
import random
from numpy.random import choice
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pyplot import cm


from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





# l = estremi degli intervalli
# data = dataframe di dati


def MondrianSupervised_SingleCut(data,t0,l,lifetime,father):


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
	
	p = np.array(ld)/np.sum(ld)
	dim = choice(d,p=p,replace=False,size=len(d))



	for d_cut in dim:
			
			
		
		if data[d_cut].max() == data[d_cut].min():
			return
		
		
		
		# cut
		#x = np.random.uniform(data[d_cut].max(),data[d_cut].min())
		clf = LogisticRegression(penalty='none').fit(np.array(data[d_cut]).reshape((len(data),1)), np.array(data['class']).reshape((len(data),1)))
		x = -clf.intercept_[0]/clf.coef_[0][0]
		
		
		
		if (x<l[d_cut][0]) or (x>l[d_cut][1]):
			continue
					
					
					
				
		l_min = l.copy()
		l_max = l.copy()
		l_min[d_cut] = [l[d_cut][0],x]
		l_max[d_cut] = [x,l[d_cut][1]]
				
	
		risultato1 = [t0, l_min]
		risultato2 = [t0, l_max]
				
				
			
				
				
		risultato = [risultato1, risultato2, x, t0, d_cut, father]
	
	


		
		return risultato
	
	return
		
		







def MondrianSupervised(X,y,t0,lifetime): 
	
	
	
	data = pd.DataFrame(X)
	data['class'] = y
	
	spazio_iniziale = []
	for i in range(len(X[0])):
		length = data[i].max() - data[i].min()
		spazio_iniziale.append([data[i].min() - length*0.05,data[i].max() + length*0.05])
	

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
			

			 
		 
			mondrian = MondrianSupervised_SingleCut(data,i[0],i[1],lifetime,i[2])
			
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
			names.append(j+str(i))
	
	
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

			









'''
p = part[part['leaf']==True].copy()


for l in data['class'].unique():
		
	dat = data[data['class']==l].copy()
	dat.index=np.arange(len(dat))
	
	for i in range(len(p)):
		for j in range(n_d):
			dat['min'+str(j)] = p['min'+str(j)].iloc[i]
			dat['max'+str(j)] = p['max'+str(j)].iloc[i]
			dat = dat.query("("+str(j)+">min"+str(j)+")")#" and ("+str(j)+"<max"+str(j)+")")
		print('part: ',i, len(dat))
		
		
		
		dat = pd.eval("dat['count'"+str(j)+"'] = (dat["+str(j)+"] > dat['min"+str(j)+"']) and (dat["+str(j)+"] < dat['max"+str(j)+"'])", inplace=True)
		dat = pd.eval("dat['count'"+str(j)+"'] = (dat["+str(j)+"] > dat['min"+str(j)+"']) and (dat["+str(j)+"] < dat['max"+str(j)+"'])", inplace=True)

'''
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
					if (dat.iloc[i][j]>=p['min'+str(j)].iloc[k]) & (dat.iloc[i][j]<p['max'+str(j)].iloc[k]):
						partial_count.append(0)
					else:
						break
				if len(partial_count) == n_d:
					count += 1
			data_count.append(count)	
		
		count_class.append(data_count)
		
		
	for i in range(len(count_class)):
		p['counts'+str(i)] = count_class[i]
		
	p.index=np.arange(len(p))	
	
	
	p = (p.eval('prob0 = (0.5 + counts0) / (1 + counts0 + counts1)')
	  .eval('prob1 = 1-prob0')	) #'prob1 = (0.5 + counts1) / (1 + counts0 + counts1)'
	
	
	p['cl'] = 0
	p_cl1 = p.query('prob1>0.5')
	p.loc[p_cl1.index,'cl']=1	

	
		
	return p









def Class(X,y,part_with_counts):	
	
	
	
	cl = []
	part_number = []
	
			#part_with_counts[str(j)+'data'] = i[j]
			#part_with_counts.eval("find_class = ("+str(j)+"data>"+str(j)+"min) and ("+str(j)+"data<"+str(j)+"max)", inplace=True)
	
	
	for i in X:
		count=0
		for j in range(len(part_with_counts)):
			count += 1
			partial_count=[]
			for k in range(len(X[0])):
				if (i[k]>=part_with_counts['min'+str(k)].iloc[j]) & (i[k]<part_with_counts['max'+str(k)].iloc[j]):
					partial_count.append(0)
				else:
					break
			if len(partial_count) == len(X[0]):
				cl.append(part_with_counts['cl'].iloc[j])  
				part_number.append(part_with_counts['part_number'].iloc[j]) 
				break
			else:
				if count==len(part_with_counts):
					cl.append('nan')
					part_number.append('nan')
					
					

	X = pd.DataFrame(X)
	X['cl_true'] = y			
	X['cl_pred'] = cl
	X['part_number_data'] = part_number
	X_bis = X.query("cl_pred!='nan'").copy()
	
	accuracy = accuracy_score(list(X_bis['cl_true']), list(X_bis['cl_pred']))
	
		
	return accuracy,X











def PartitionPlot3D(X,y,part):
	
	
	data = pd.DataFrame(X)
	data['class'] = y

	
	p = part[part['leaf']==True]
	
	p = p[['min0','max0','min1','max1','min2','max2']]

	p['indice'] = np.arange(1,len(p)+1,1)
	
	
	p = (p.eval("min_r0 = min0 + (max0-min0)*0.05")
		 .eval("min_r1 = min1 + (max1-min1)*0.05")
		 .eval("min_r2 = min2 + (max2-min2)*0.05")
		 .eval("max_r0 = max0 - (max0-min0)*0.05")
		 .eval("max_r1 = max1 - (max1-min1)*0.05")
		 .eval("max_r2 = max2 - (max2-min2)*0.05")
		 )

	
	
	h1 = [['min_r','max_r'],
		  ['min_r','max_r'],
	      ['min_r','max_r'],
	      ['min_r','max_r']]
	h1 = pd.DataFrame(h1)
	
	
	
	h2 = [['min_r','min_r'],
	      ['min_r','min_r'],
	      ['max_r','max_r'],
	      ['max_r','max_r']]
	h2 = pd.DataFrame(h2)
	h2.columns = [2,3]
	
	
	
	h3 = [['min_r','min_r'],
	      ['max_r','max_r'],
	      ['min_r','min_r'],
	      ['max_r','max_r']]
	h3 = pd.DataFrame(h3)
	h3.columns = [4,5]
	
	
	h = pd.concat([h1,h2,h3],axis=1)
	
	order = [[0, 1, 2, 3, 4, 5],[2, 3, 0, 1, 4, 5],[2, 3, 4, 5, 0, 1]]
	
	
	
	
	
	color=cm.rainbow(np.linspace(0,1,len(p)))
	
	ax = plt.axes(projection='3d')
	
	
	for i,c in zip(range(len(p)),color):
		for j in range(len(h)):
			for k in order:
				x_min = p[h[k[0]].iloc[j]+'0'].iloc[i]
				x_max = p[h[k[1]].iloc[j]+'0'].iloc[i]
				y_min = p[h[k[2]].iloc[j]+'1'].iloc[i]
				y_max = p[h[k[3]].iloc[j]+'1'].iloc[i]
				z_min = p[h[k[4]].iloc[j]+'2'].iloc[i]
				z_max = p[h[k[5]].iloc[j]+'2'].iloc[i]
				
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
		
		ax.vlines(p['min0'].iloc[i],p['min1'].iloc[i],p['max1'].iloc[i])		
		ax.vlines(p['max0'].iloc[i],p['min1'].iloc[i],p['max1'].iloc[i])
		ax.hlines(p['min1'].iloc[i],p['min0'].iloc[i],p['max0'].iloc[i])
		ax.hlines(p['max1'].iloc[i],p['min0'].iloc[i],p['max0'].iloc[i])
		ax.text(p['min0'].iloc[i],p['min1'].iloc[i],p['part_number'].iloc[i])

		
		
		
	ax.scatter(data[data['class']==0][0],data[data['class']==0][1])
	ax.scatter(data[data['class']==1][0],data[data['class']==1][1])
	
	
	
	plt.show()
	
	return
	
	



def PartitionPlot(X,y,part):
	
	if len(X[0]) == 2:
		PartitionPlot2D(X,y,part)
		
	if len(X[0]) == 3:
		PartitionPlot3D(X,y,part)
		
		
		
	return
		








