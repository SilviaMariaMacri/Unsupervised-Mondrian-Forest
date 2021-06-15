import numpy as np
from numpy.random import choice
import pandas as pd


from sklearn.linear_model import LogisticRegression
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
		clf = LogisticRegression(penalty='none').fit(np.array(data[d_cut]).reshape((len(data),1)), np.array(data['class']))
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

			










def Count(X,y,part): 


	data = pd.DataFrame(X)
	n_d = len(data.columns)
	data.columns = [*[f'coord_{s}' for s in data.columns]]
	data['class'] = y
	
	
	
	p = part[part['leaf']==True].copy()
	p.index = np.arange(len(p))
	
	
	for l in data['class'].unique():
		
		count = []
		for i in range(len(p)):
			dat = data[data['class']==l].copy()
			dat.index=np.arange(len(dat))
	
			for j in range(n_d):
				dat['min'+str(j)] = p['min'+str(j)].iloc[i]
				dat['max'+str(j)] = p['max'+str(j)].iloc[i]
				dat.eval('higher_min'+str(j)+' = coord_'+str(j)+'> min'+str(j), inplace=True)
				dat.eval('lower_max'+str(j)+' = coord_'+str(j)+'< max'+str(j), inplace=True)
				dat = dat.query('(higher_min'+str(j)+'==True) and (lower_max'+str(j)+'==True)').copy()
				
			count.append(len(dat))	
		
		p['count'+str(l)] = count
		
	
	#funziona per classificazione binaria	
	p = (p.eval('prob0 = (0.5 + count0) / (1 + count0 + count1)')
	  .eval('prob1 = 1-prob0')	) #'prob1 = (0.5 + counts1) / (1 + counts0 + counts1)'
	
	
	p['cl'] = 0
	p_cl1 = p.query('prob1>0.5')
	p.loc[p_cl1.index,'cl']=1	

	
		
	return p



		



def AssignClass(X,y,part_with_counts):	
	
	
	
	d = np.arange(len(X[0]))
	
	
	part_number = []

	
	for i in X:

		p = part_with_counts.copy()
		
		for j in d:
			p['data'+str(j)] = i[j]
			p = p.query("(data"+str(j)+">=min"+str(j)+") & (data"+str(j)+"<max"+str(j)+")").copy()
		
		try:
			part_number.append(p['part_number'].iloc[0])
		except IndexError:
			part_number.append('nan')


	X = pd.DataFrame(X)
	X['part_number'] = part_number
	
	X = pd.merge(X,part_with_counts[['part_number','cl']],left_on='part_number',right_on='part_number')
	
	X['cl_true'] = y

	X_bis = X.query("cl!='nan'").copy()
	
	accuracy = accuracy_score(list(X_bis['cl_true']), list(X_bis['cl']))
	
	

	
	return accuracy,X


