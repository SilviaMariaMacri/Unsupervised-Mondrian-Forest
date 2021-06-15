import numpy as np
from numpy.random import choice
import pandas as pd	
from scipy.spatial import distance

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.pyplot import cm



#%%

# coppie di vertici 
#vertici_iniziali=[ [[0,0],[1,0]], [[1,0],[1,1]], [[1,1],[0,1]], [[0,1],[0,0]]]
t0=0
lifetime=3

m,box,df=MondrianPolygon(X,t0,lifetime)



#%%

sns.set_style('whitegrid')
fig,ax = plt.subplots()

color=cm.rainbow(np.linspace(0,1,len(box)))
for i,c in zip(range(len(box)),color):
	p = Polygon(box[i], facecolor = 'none', edgecolor=c)
	ax.add_patch(p)
	
	ax.scatter(m[i][2][0],m[i][2][1],color=c)
	
xmin = box[0][0][0]-0.05
ymin = box[0][0][1]-0.05
xmax = box[0][2][0]+0.05
ymax = box[0][2][1]+0.05
	
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)

#dati = pd.DataFrame(X)
#ax.scatter(dati[0],dati[1])

plt.show()




#%% singola partizione con dati corrispondenti




for i in range(len(box)):
	
	sns.set_style('whitegrid')
	fig,ax = plt.subplots()
	
	p = Polygon(box[i], facecolor = 'none')
	ax.add_patch(p)
	ax.scatter(m[i][2][0],m[i][2][1])
	
	plt.show()

	

#%%  area poligono


box = box[0]


def PolyArea(box):
	
	df_box = pd.DataFrame(box)
	x = df_box[0]
	y = df_box[1]
    
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

print(PolyArea(box))



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






def trova_part_vicine(part):


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
	df.index = np.arange(len(df))
	



	return df
# per più di due dimensioni?
# come generalizzarla a partizione con tagli non regolari?





#  calcolo varianza per partizioni vicine 


def calcolo_varianza_part_vicine(data,i,j):
	
	
	
	data1 = data.query('part_number=='+str(i))
	data2 = data.query('part_number=='+str(j))
	
	pd1 = pdist(data1)
	pd2 = pdist(data2)
	
	pd = pdist(data)
	pd12 = np.hstack([pd1, pd2])
	
	var_part_unica = np.var(pd)
	var1 = np.var(pd1)
	var2 = np.var(pd2)
	
	#var_part_separate = np.var(pd12)
	#score_1 = np.abs(np.log(np.var(pd12)/np.var(pd)))
	#score_2 = np.abs(np.log(np.var(pd1)/np.var(pd2)))
	#print(score_1>score_2)

	
	return var_part_unica,var1,var2 #score_1,score_2, var_part_unica,var_part_separate





part_vicine = trova_part_vicine(part)
punti = AssignPartition(X,part)

#separazione_corretta = [] 


part1 = []
part2 = []
score_1 = []
score_2 = []
#v_unica = []
#v_sep = []


for i in part_vicine['part_number']:
	
	for j in list(part_vicine[part_vicine['part_number']==i]['neighbors'])[0]:
		part2.append(j)
		part1.append(i)
		
		p = punti.query('(part_number=='+str(i)+') or (part_number=='+str(j)+')').copy()
		
		
		var_part_unica,var1,var2 = calcolo_varianza_part_vicine(punti,i,j)
		#s1,s2,v1,v2 = calcolo_varianza_part_vicine(punti,i,j)
		#score_1.append(s1)
		#score_2.append(s2)
		#v_unica.append(v1)
		#v_sep.append(v2)
		
		
df = {'part1':part1,'part2':part2,'var_part_unica':var_part_unica,'var1':var1,'var2':var2}#'score_1':score_1,'score_2':score_2}		
df = pd.DataFrame(df)		#'v_unica':v_unica,'v_sep':v_sep}#
		#if s1 > s2:
		#	separazione_corretta.append(True)
		#else:
		#	separazione_corretta.append(False)
		

		
		
	
#conto_punti = punti.query('part_number=='+str(i)).count()[0]


#%%



import numpy as np
import pylab as plt
import scipy.stats as st
from scipy.spatial.distance import cdist, pdist
# %%
N = 50
for diff in np.linspace(-1, 1, 21):
    results_pdc = []
    results_pdt = []
    results_ratio = []
    results_ratio_internal = []
    results = []
    for i in range(256):
        p1 = plt.randn(N, 2) - diff
        p2 = plt.randn(N, 2) + diff
        pt = np.vstack([p1, p2])
        pd1 = pdist(p1)
        pd2 = pdist(p2)
        pdc = np.hstack([pd1, pd2])
        pdt = pdist(pt)
        results_pdt.append(np.var(pdt))
        results_pdc.append(np.var(pdc))
        score_1 = np.abs(np.log(np.var(pdc)/np.var(pdt)))
        #score_1 = np.abs(np.var(pdc)/np.var(pdt))
		#print(score_1)
        results_ratio.append(score_1) 
        score_2 = np.abs(np.log(np.var(pd1)/np.var(pd2)))
        #score_2 = np.abs(np.var(pd1)/np.var(pd2))
        results_ratio_internal.append(score_2)
        results.append(score_1>score_2)
	
	#print('score1: ',np.mean(results_ratio))
	#print('score2: ',np.mean(results_ratio_internal))		
    print(diff.round(1), np.mean(results))

# %%


N = 50
for diff in np.linspace(-1, 1, 21):
    results_pdc = []
    results_pdt = []
    results_ratio = []
    results_ratio_internal = []
    results = []
    for i in range(256):
        p1 = plt.randn(N, 2) - diff
        p2 = plt.randn(N, 2) + diff
        pt = np.vstack([p1, p2]) # unico gruppo
        pd1 = pdist(p1)
        pd2 = pdist(p2)
        pdc = np.hstack([pd1, pd2])
        pdt = pdist(pt)
        results_pdt.append(np.var(pdt))
        results_pdc.append(np.var(pdc))
        score_1 = np.abs(np.log(np.var(pdc)/np.var(pdt)))
        results_ratio.append(score_1)
        score_2 = np.abs(np.log(np.var(pd1)/np.var(pd2)))
        results_ratio_internal.append(score_2)
        results.append(score_1>score_2)
    print(diff.round(0), np.mean(results))	
	

	

# %%
fig, axes = plt.subplot_mosaic([["confronto"], ["log ratio"]])
axes["confronto"].boxplot([results_pdt, results_pdc], notch=True)
axes["log ratio"].boxplot([results_ratio, results_ratio_internal], notch=True)

