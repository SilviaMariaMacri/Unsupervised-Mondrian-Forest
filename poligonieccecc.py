#partizionamento non perpendicolare a direzioni
#associare partizioni vicine
#confrontare partizioni vicine in base a distanza



import numpy as np
from numpy.random import choice
import pandas as pd	
import random
from scipy.spatial import distance

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon




#%%

# coppie di vertici
vertici_iniziali=[ [[0,0],[1,0]], [[1,0],[1,1]], [[1,1],[0,1]], [[0,1],[0,0]]]
t0=0
lifetime=1

m,box,df=MondrianPolygon(t0,vertici_iniziali,lifetime)


#%%


fig,ax = plt.subplots()

for i in box:
	p = Polygon(i, facecolor = 'none', edgecolor='b')
	ax.add_patch(p)


#ax.set_xlim(-0.5,1.5)
#ax.set_ylim(-0.5,1.5)

plt.show()


#%%



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





#%%       trovare partizioni vicine tagli perpendicolari


neighbors = []

for i in range(len(part.query('leaf==True'))):
	
	p = part.query('leaf==True').copy()
	p.index = np.arange(len(part.query('leaf==True')))
	
	for j in range(2):
		p['min'+str(j)+'_'+str(i)] = p['min'+str(j)].iloc[i]
		p['max'+str(j)+'_'+str(i)] = p['max'+str(j)].iloc[i]
		
	for j in range(2):	
		p=(p.eval('vicinoA'+str(j)+'_'+str(i)+' = ((min'+str(j)+'>=min'+str(j)+'_'+str(i)+') and (min'+str(j)+'<=max'+str(j)+'_'+str(i)+')) or ( (max'+str(j)+'>=min'+str(j)+'_'+str(i)+') and (max'+str(j)+'<=max'+str(j)+'_'+str(i)+'))')
	 .eval('vicinoB'+str(j)+'_'+str(i)+' = ((min'+str(j)+'_'+str(i)+'>=min'+str(j)+') and (min'+str(j)+'_'+str(i)+'<=max'+str(j)+')) or ( (max'+str(j)+'_'+str(i)+'>=min'+str(j)+') and (max'+str(j)+'_'+str(i)+'<=max'+str(j)+'))'))
		
		#p=p.query('vicino'+str(j)+'_'+str(i)+'==True')
		p=p.query('(vicinoA'+str(j)+'_'+str(i)+'==True) or (vicinoB'+str(j)+'_'+str(i)+'==True)')
		
		
	p = p.drop(i)
	
	neighbors.append(list(p['part_number']))



df={'part_number':part.query('leaf==True')['part_number'],'neighbors':neighbors}
df=pd.DataFrame(df)



# per più di due dimensioni?
# come generalizzarla a partizione con tagli nonr egolari?




#%%
# quella cosa del confronto distanze fra partizioni vicine e all'interno 
#della stessa partizione funziona solo con griglia regolare



punti = AssignPartition(X,part)

punti.query('part_number==41')[0].max()









