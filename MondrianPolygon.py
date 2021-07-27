import time

start = time.time()
end = time.time()
print(end - start)




#%%

import numpy as np
from numpy.random import choice
import pandas as pd	
from scipy.spatial import distance
from itertools import combinations
#import matplotlib.pyplot as plt
#from matplotlib.patches import Polygon
from scipy.spatial.distance import pdist








def	Cut_without_data(l,ld):

	
	n_vert = np.arange(len(l))
		
	p=ld/sum(np.array(ld))
	lati = choice(n_vert,p=p,replace=False,size=2)
	lati.sort()
		
	point1 = np.array(l[lati[0]][0]) + (np.array(l[lati[0]][1]) - np.array(l[lati[0]][0]))*np.random.uniform(0,1)
	point2 = np.array(l[lati[1]][0]) + (np.array(l[lati[1]][1]) - np.array(l[lati[1]][0]))*np.random.uniform(0,1)

	return point1,point2,lati




#assegna i dati alla partizione corrispondente a partire dalle coordinate dei punti
#di intersezione e dalla matrice
def FindDataPartition2D_vecchia(points,matrix):


	cut = [matrix['norm_vect_0'].iloc[0], matrix['norm_vect_1'].iloc[0], matrix['magnitude_norm_vect'].iloc[0]] 

	
	
	# quadrante 1 
	if (cut[0]>0 and cut[1]>0) or (cut[0]<0 and cut[1]>0):
		if points['x'].iloc[0] > points['x'].iloc[1]:
			data1 = matrix.query('dist_point_cut<0')[['point_0','point_0','point_index']].copy()
			data2 = matrix.query('dist_point_cut>0')[['point_0','point_1','point_index']].copy()
		else:
			data2 = matrix.query('dist_point_cut<0')[['point_0','point_1','point_index']].copy()
			data1 = matrix.query('dist_point_cut>0')[['point_0','point_1','point_index']].copy()
	

		#xmax,xmin in ordine:
		#	l1 = dist - 
		#	l2 = dist +
		#xmin,xmax:
		#	l1 = dist +
		#	l2 = dist -
		
	# quadrante 3
	if (cut[0]<0 and cut[1]<0) or (cut[0]>0 and cut[1]<0):
		if points['x'].iloc[0] > points['x'].iloc[1]:
			data2 = matrix.query('dist_point_cut<0')[['point_0','point_1','point_index']].copy()
			data1 = matrix.query('dist_point_cut>0')[['point_0','point_1','point_index']].copy()
		else:
			data1 = matrix.query('dist_point_cut<0')[['point_0','point_1','point_index']].copy()
			data2 = matrix.query('dist_point_cut>0')[['point_0','point_1','point_index']].copy()


		#xmax,xmin in ordine:
		#	l1 = dist + 
		#	l2 = dist -
		#xmin,xmax:
		#	l1 = dist -
		#	l2 = dist +



	data1.index = np.arange(len(data1))
	data1.columns = [0,1,'index']
	data2.index = np.arange(len(data2))
	data2.columns = [0,1,'index']
	
	
	
	return data1,data2










def CutCoeff_DistMax(dist_matrix):

	# a*x + b*y = c      retta cut
	matrix = dist_matrix[dist_matrix['dist']==dist_matrix['dist'].max()].copy()
	matrix.index = np.arange(len(matrix))
	a = matrix['norm_vect_0'].iloc[0] #versore0
	b = matrix['norm_vect_1'].iloc[0] #versore1
	c = matrix['magnitude_norm_vect'].iloc[0] #modulo   
	

	return a,b,c,matrix







def Variance(data1,data2,data):
	
	
	
	#if (len(data1)>2) & (len(data2)>2):
	if (len(data1)!=1) & (len(data2)!=1):	
	
		pd1 = pdist(data1)
		pd2 = pdist(data2)
		
		#data = np.vstack([data1,data2])
		pd = pdist(data)
		pd12 = np.hstack([pd1, pd2])
		
		var_ratio = np.var(pd)/np.var(pd12) 
				
		return var_ratio 
		
	else:
		s='nan'
		return s







def CutVariance(dist_matrix,data):
	
	data = data.drop([0,1],axis=1)
	data_pair = list(combinations(data['index'], 2))
	data_pair = pd.DataFrame(data_pair)
	data_pair.columns = ['index2','index1']


	var_ratio = []
	for i in range(len(data_pair)):
		
		matrix = dist_matrix[(dist_matrix['index1']==data_pair['index1'].iloc[i]) & (dist_matrix['index2']==data_pair['index2'].iloc[i])].copy()
		
		data12 = matrix[['point_0','point_1']].copy()
		data1 = matrix.query('dist_point_cut>0')[['point_0','point_1']].copy()
		data2 = matrix.query('dist_point_cut<0')[['point_0','point_1']].copy()
		
		var = Variance(data1,data2,data12)
		var_ratio.append(var)
		
		
	data_pair['var_ratio'] = var_ratio

	if data_pair['var_ratio'].unique()[0]=='nan':
		return 

	# cancella righe con var_ratio=nan 
	data_pair = data_pair.drop(data_pair[data_pair['var_ratio']=='nan'].index)
	data_pair.index = np.arange(len(data_pair))

	q=data_pair['var_ratio']**150
	p = q/q.sum()
	index_cut = choice(data_pair.index,p=p)
	
	# a*x + b*y = c  retta cut
	matrix = dist_matrix[(dist_matrix['index1']==data_pair['index1'].iloc[index_cut]) & (dist_matrix['index2']==data_pair['index2'].iloc[index_cut])].copy()
	matrix.index = np.arange(len(matrix))
	a = matrix['norm_vect_0'].iloc[0] #versore0
	b = matrix['norm_vect_1'].iloc[0] #versore1
	c = matrix['magnitude_norm_vect'].iloc[0] #modulo 
	
	
	return a,b,c,matrix  




def Centroid(data1,data2,data12):

	
	data_tot = [data1,data2,data12]
	centr=[[],[],[]]
	for j in range(3):
		for i in range(len(data12[0])):
			centr[j].append(np.mean(data_tot[j][:,i]))
	'''	
	mean_dist=[]
	for i in range(3):
		mean_dist.append(np.mean(cdist(data_tot[i],[centr[i]])))
	
	#ratio = mean_dist[2]/np.mean(mean_dist[0:2])	
	ratio = mean_dist[2]/(mean_dist[0] + mean_dist[1])	
	'''
	
	dist=[]
	for i in range(3):
		dist.append(cdist(data_tot[i],[centr[i]]))
		
	ratio = np.mean(dist[2])/np.mean(np.vstack([dist[0],dist[1]]))	
	difference = np.mean(dist[2]) - np.mean(np.vstack([dist[0],dist[1]]))
	
	
	return ratio,difference	  	  

	  



def CutCentroid(dist_matrix,data):
	
	data = data.drop([0,1],axis=1)
	data_pair = list(combinations(data['index'], 2))
	data_pair = pd.DataFrame(data_pair)
	data_pair.columns = ['index2','index1']


	ratio_mean_dist = []
	difference_mean_dist = []
	for i in range(len(data_pair)):
		
		
		matrix = dist_matrix[(dist_matrix['index1']==data_pair['index1'].iloc[i]) & (dist_matrix['index2']==data_pair['index2'].iloc[i])].copy()
		
		data12 = np.array(matrix[['point_0','point_1']].copy())
		data1 = np.array(matrix.query('dist_point_cut>0')[['point_0','point_1']].copy())
		data2 = np.array(matrix.query('dist_point_cut<0')[['point_0','point_1']].copy())
		
		if (len(data1)==1) or (len(data2)==1):
			ratio_mean_dist.append('nan')
			difference_mean_dist.append('nan')
		else:
			ratio,difference = Centroid(data1,data2,data12)
			ratio_mean_dist.append(ratio)
			difference_mean_dist.append(difference)
		
	data_pair['ratio_centroid'] = ratio_mean_dist
	data_pair['difference_centroid'] = difference_mean_dist


	if data_pair['ratio_centroid'].unique()[0]=='nan':
		return 
	if data_pair['difference_centroid'].unique()[0]=='nan':
		return 


	# cancella righe con ratio e difference=nan FORSE QUESTO LO DEVI CAMBIARE A SECONDA CHE SI USI UNOO O L'ALTRO
	#data_pair = data_pair.drop(data_pair[data_pair['ratio_centroid']=='nan'].index)
	data_pair = data_pair.drop(data_pair[data_pair['difference_centroid']=='nan'].index)
	
	data_pair.index = np.arange(len(data_pair))

	q=data_pair['difference_centroid']**100
	#q=data_pair['ratio_centroid']**100
	p = q/q.sum()
	index_cut = choice(data_pair.index,p=p)
	
	# a*x + b*y = c  retta cut
	matrix = dist_matrix[(dist_matrix['index1']==data_pair['index1'].iloc[index_cut]) & (dist_matrix['index2']==data_pair['index2'].iloc[index_cut])].copy()
	matrix.index = np.arange(len(matrix))
	a = matrix['norm_vect_0'].iloc[0] #versore0
	b = matrix['norm_vect_1'].iloc[0] #versore1
	c = matrix['magnitude_norm_vect'].iloc[0] #modulo 


	return a,b,c,matrix 






def Cut(dist_matrix,data):
	

	data_index = data.drop([0,1],axis=1)
	data_pair = list(combinations(data_index['index'], 2))
	data_pair = pd.DataFrame(data_pair)
	data_pair.columns = ['index2','index1']


	df = pd.DataFrame() 
	for i in range(len(data_pair)):
		#print(i)
		# considero interesezioni di tutti i punti con una distanza fra due punti in particolare
		matrix = dist_matrix.query('(index1=='+str(data_pair['index1'].iloc[i])+' and index2=='+str(data_pair['index2'].iloc[i])+') or (index1=='+str(data_pair['index2'].iloc[i])+' and index2=='+str(data_pair['index1'].iloc[i])+')').copy()
		# considero solo proiezioni che cadono all'interno del segmento
		matrix = matrix.query('(intersez_0>='+str(min(matrix['x1_0'].iloc[0],matrix['x2_0'].iloc[0]))+') and (intersez_0<='+str(max(matrix['x1_0'].iloc[0],matrix['x2_0'].iloc[0]))+')')
		# ordino i punti di intersezione
		matrix = matrix.sort_values(by='intersez_0')
		matrix.index = np.arange(len(matrix))
		
		point_min = np.array(matrix[['intersez_0','intersez_1']][:-1])
		point_max = np.array(matrix[['intersez_0','intersez_1']][1:])
		dist_punti_consecutivi = list(np.diag(cdist(point_min,point_max)))
		
		#point_max = list(point_max)
		#point_max.append(['nan','nan'])
		#matrix[['punto_adiac_0','punto_adiac_1']] = point_max
		dist_punti_consecutivi.append('nan')
		#indice_punto_adiacente = list(matrix['point_index'].iloc[1:])
		#indice_punto_adiacente.append('nan')
		#matrix['point_index_successivo'] = indice_punto_adiacente
		
		intersez_adiacente_0 = list(matrix['intersez_0'].iloc[1:])
		intersez_adiacente_0.append('nan')
		intersez_adiacente_1 = list(matrix['intersez_1'].iloc[1:])
		intersez_adiacente_1.append('nan')
		matrix['intersez_0_successivo'] = intersez_adiacente_0
		matrix['intersez_1_successivo'] = intersez_adiacente_1
		
		matrix['dist_punti_consecutivi'] = dist_punti_consecutivi
		# media senza valore assoluto?
		matrix['media_distanze'] = np.mean(matrix['dist_point_dist'])
		
	
		df = df.append(matrix[matrix['dist_punti_consecutivi']==matrix['dist_punti_consecutivi'].iloc[:-1].max()])
		
		#if len(matrix)<=1:
		#	df = df.append(dist_matrix.query('(index1=='+str(data_pair['index1'].iloc[i])+' and index2=='+str(data_pair['index2'].iloc[i])+') or (index1=='+str(data_pair['index2'].iloc[i])+' and index2=='+str(data_pair['index1'].iloc[i])+')').iloc[0])
	
	df.index = np.arange(len(df))
	
	
	
	q=df['dist_punti_consecutivi']**50
	p = q/q.sum()
	index_cut = choice(df.index,p=p)
	
	# a*x + b*y = c  retta cut
	point_cut_0 = (df['intersez_0'].iloc[index_cut]+df['intersez_0_successivo'].iloc[index_cut])/2
	point_cut_1 = (df['intersez_1'].iloc[index_cut]+df['intersez_1_successivo'].iloc[index_cut])/2
	 
	a = df['norm_vect_cut_0'].iloc[index_cut] #versore0
	b = df['norm_vect_cut_1'].iloc[index_cut] #versore1
	c = np.dot([point_cut_0,point_cut_1],list(df.iloc[index_cut][['norm_vect_cut_0', 'norm_vect_cut_1']]))
	
	#['norm_vect_cut_0', 'norm_vect_cut_1','magnitude_norm_point', 'intersez_0',
    #+'intersez_1', 'intersez_0_successivo','intersez_1_successivo', 'dist_punti_consecutivi']

	matrix = dist_matrix[(dist_matrix['index1']==df['index1'].iloc[index_cut]) & (dist_matrix['index2']==df['index2'].iloc[index_cut])].copy()
	matrix.index = np.arange(len(matrix))
	
	############
	versori_iperpiani = np.array(df[['norm_vect_cut_0', 'norm_vect_cut_1']].iloc[index_cut])
	magnitude_norm_vect = versori_iperpiani@(np.array([point_cut_0,point_cut_1]).T)
	matrix['magnitude_norm_vect'] = magnitude_norm_vect
	
	X = np.array(matrix[['point_0','point_1']])
	dist_point_cut = versori_iperpiani@X.T - matrix['magnitude_norm_vect']
	matrix['dist_point_cut'] = dist_point_cut

	return a,b,c,matrix #(credo)





# vale solo in 2D 
def Cut_with_data(data,l,dist_matrix):
	
	
	n_vert = np.arange(len(l))
	

	#scegline uno
	#a,b,c,matrix = CutCoeff_DistMax(dist_matrix)
	try:
		a,b,c,matrix = CutVariance(dist_matrix,data)
		#a,b,c,matrix = CutCentroid(dist_matrix,data)
		#a,b,c,matrix = Cut(dist_matrix,data)
	except TypeError:
		return
	
	
	coordx = [] #coordinata x
	coordy = [] #coord y
	lati = []
	#ax + by = c
	for i in n_vert:
		# (y1-y2)*(x-x2) = (x1-x2)*(y-y2)  retta passante per i due vertici
		al = l[i][0][1] - l[i][1][1] # y1-y2
		bl = l[i][1][0] - l[i][0][0] # x2-x1
		cl = l[i][0][1]*l[i][1][0] - l[i][0][0]*l[i][1][1] # y1*x2-x1*y2
		
		A = [[al,bl],[a,b]]
		Ax = [[cl,bl],[c,b]]
		Ay = [[al,cl],[a,c]]
		
		detA = np.linalg.det(A)
		detAx = np.linalg.det(Ax)
		detAy = np.linalg.det(Ay)
		
		# coordinate intersezione
		x = detAx/detA
		y = detAy/detA
		
		if (round(x,6)<min(round(l[i][0][0],6),round(l[i][1][0],6))) or  (round(x,6)>max(round(l[i][0][0],6),round(l[i][1][0],6))) or (round(y,6)<min(round(l[i][0][1],6),round(l[i][1][1],6))) or  (round(y,6)>max(round(l[i][0][1],6),round(l[i][1][1],6))):
		#if (x<min(l[i][0][0],l[i][1][0])) or  (x>max(l[i][0][0],l[i][1][0])) or (y<min(l[i][0][1],l[i][1][1])) or  (y>max(l[i][0][1],l[i][1][1])):
			continue
		
		coordx.append(x)
		coordy.append(y)
		lati.append(i)


	points = {'lati':lati,'x':coordx,'y':coordy}
	points = pd.DataFrame(points)	
	
	
	point1 = list(points[['x','y']].iloc[0])
	point2 = list(points[['x','y']].iloc[1])
	lati = list(points['lati'])


	#data1,data2 = FindDataPartition2D_vecchia(points,matrix)

	
	
	return point1,point2,lati,matrix#,data1,data2




















def Cut_with_data_griglia_regolare(data,l,dist_matrix):
	
	
	n_vert = np.arange(len(l))


	for j in range(len(dist_matrix['index_norm_vect'].unique())):
		
		print(j)

		matrix = dist_matrix[dist_matrix['index_norm_vect']==dist_matrix['index_norm_vect'].unique()[j]].copy()
		matrix.index = np.arange(len(matrix))
		
		a = matrix['norm_vect_0'].iloc[0] #versore0 
		b = matrix['norm_vect_1'].iloc[0] #versore1
		c = matrix['magnitude_norm_vect'].iloc[0] #modulo 




		coordx = [] #coordinata x
		coordy = [] #coord y
		lati = []
		#ax + by = c
		for i in n_vert:
			# (y1-y2)*(x-x2) = (x1-x2)*(y-y2)  retta passante per i due vertici
			al = l[i][0][1] - l[i][1][1] # y1-y2
			bl = l[i][1][0] - l[i][0][0] # x2-x1
			cl = l[i][0][1]*l[i][1][0] - l[i][0][0]*l[i][1][1] # y1*x2-x1*y2
			
			A = [[al,bl],[a,b]]
			Ax = [[cl,bl],[c,b]]
			Ay = [[al,cl],[a,c]]
			
			detA = np.linalg.det(A)
			detAx = np.linalg.det(Ax)
			detAy = np.linalg.det(Ay)
			
			# coordinate intersezione
			x = detAx/detA
			y = detAy/detA
			
		#	if type(x)!=int:
		#		continue
			
			if (round(x,6)<min(round(l[i][0][0],6),round(l[i][1][0],6))) or  (round(x,6)>max(round(l[i][0][0],6),round(l[i][1][0],6))) or (round(y,6)<min(round(l[i][0][1],6),round(l[i][1][1],6))) or  (round(y,6)>max(round(l[i][0][1],6),round(l[i][1][1],6))):
			#if (x<min(l[i][0][0],l[i][1][0])) or  (x>max(l[i][0][0],l[i][1][0])) or (y<min(l[i][0][1],l[i][1][1])) or  (y>max(l[i][0][1],l[i][1][1])):
				continue
			
			coordx.append(x)
			coordy.append(y)
			lati.append(i)


		#if len(coordx)==2:
		#	break
		
		

	points = {'lati':lati,'x':coordx,'y':coordy}
	points = pd.DataFrame(points)	
	
	
	point1 = list(points[['x','y']].iloc[0])
	point2 = list(points[['x','y']].iloc[1])
	lati = list(points['lati'])


	#data1,data2 = FindDataPartition2D_vecchia(points,matrix)

	
	
	return point1,point2,lati,matrix#,data1,data2












def FindDataPartition2D(matrix,l1,l2):
	
	
	
	names = []
	for i in range(len(l1[0][0])):
		names.append('norm_vect_'+str(i))
		#names.append('norm_vect_cut_'+str(i)) # solo per taglio con proiezioni ecc ecc


	#distanza vertice da taglio
	d = np.dot(matrix[names].iloc[0],l1[1][1]) - matrix['magnitude_norm_vect'].iloc[0] 
	if d>0:
		data1 = matrix.query('dist_point_cut>0')[['point_0','point_1','point_index']].copy()
		data2 = matrix.query('dist_point_cut<0')[['point_0','point_1','point_index']].copy()
	else:
		data1 = matrix.query('dist_point_cut<0')[['point_0','point_1','point_index']].copy()
		data2 = matrix.query('dist_point_cut>0')[['point_0','point_1','point_index']].copy()
		

	data1.index = np.arange(len(data1))
	data1.columns = [0,1,'index']
	data2.index = np.arange(len(data2))
	data2.columns = [0,1,'index']
	
	
	
	return data1,data2











#i[2],i[0],i[1],lifetime,i[3],dist_matrix,i[4]
def MondrianPolygon_SingleCut(data,t0,l,lifetime,father,dist_matrix,neighbors):

	

	# array di lunghezze intervalli
	ld = []
	for i in l:
		ld.append(distance.euclidean(i[0],i[1]))

		
	# linear dimension
	LD = sum(ld)  
	
	

	# genera tempo per cut 
	time_cut = np.random.exponential(1/LD)


	t0 += time_cut
	
	if t0 > lifetime:
		return
	
	
	if len(data) <= 2: 
		return

	

	#senza dati		
	#point1,point2,lati = Cut_without_data(l,ld)
	
	
	
	#con dati
	dist_matrix = pd.merge(dist_matrix,data[data['index']!=dist_matrix['index1'].unique()[len(dist_matrix['index1'].unique())-1]], how='right', left_on='index2', right_on='index')
	dist_matrix = dist_matrix.drop([0,1,'index'],axis=1)
	dist_matrix = pd.merge(dist_matrix,data[data['index']!=0], how='right', left_on='index1', right_on='index')
	dist_matrix = dist_matrix.drop([0,1,'index'],axis=1)
	dist_matrix = pd.merge(dist_matrix,data, how='right', left_on='point_index', right_on='index')
	dist_matrix = dist_matrix.drop([0,1,'index'],axis=1)
	
	try:
		point1,point2,lati,matrix = Cut_with_data(data,l,dist_matrix) #,data1,data2
	except TypeError:
		return

	
	



	#l1 = l.copy()
	#l2 = l.copy()		
	
	l1 = []
	l1.append([point1,point2])
	neigh1 = []
	neigh1.append('to_update')
	for i in range(lati[1],len(l)):
		if i == lati[1]:
			l1.append([point2,l[lati[1]][1]])
		else:
			l1.append(l[i])
		neigh1.append(neighbors[i])
				
	for i in range(lati[0]+1):
		if i == lati[0]:
			l1.append([l[lati[0]][0],point1])
		else:
			l1.append(l[i])
		neigh1.append(neighbors[i])
			
			
			
	l2 = []
	l2.append([point2,point1])
	neigh2 = []
	neigh2.append('to_update')
	for i in range(lati[0],lati[1]+1):
		if i == lati[0]:
			l2.append([point1,l[lati[0]][1]])
		if i == lati[1]:
			l2.append([l[lati[1]][0],point2])
		if (i != lati[0]) & (i != lati[1]):
			l2.append(l[i])
		neigh2.append(neighbors[i])
	
	
	
	data1,data2 = FindDataPartition2D(matrix,l1,l2)
	
	
	
	
	part1 = [t0, l1, data1, neigh1]
	part2 = [t0, l2, data2, neigh2]
	risultato = [part1, part2, t0, father]
	
	
	
	
	return risultato







def MondrianPolygon(X,t0,lifetime,dist_matrix):



	#dist_matrix = DistanceMatrix(X)
	
	data = pd.DataFrame(X)
	data['index'] = data.index
	
	# 2D
	vertici_iniziali = []
	length0 = data[0].max() - data[0].min()
	length1 = data[1].max() - data[1].min()
	vertici_iniziali.append([[data[0].min() - length0*0.05,data[1].min() - length1*0.05],[data[0].max() + length0*0.05,data[1].min() - length1*0.05]])
	vertici_iniziali.append([[data[0].max() + length0*0.05,data[1].min() - length1*0.05],[data[0].max() + length0*0.05,data[1].max() + length1*0.05]])
	vertici_iniziali.append([[data[0].max() + length0*0.05,data[1].max() + length1*0.05],[data[0].min() - length0*0.05,data[1].max() + length1*0.05]])
	vertici_iniziali.append([[data[0].min() - length0*0.05,data[1].max() + length1*0.05],[data[0].min() - length0*0.05,data[1].min() - length1*0.05]])



	
	#neighbors = []
	#neighbors_i = []
	#for i in vertici_iniziali:
	#	neighbors_i.append('nan')
	#neighbors.append(neighbors_i)
	neighbors = []
	for i in vertici_iniziali:
		neighbors.append(np.nan)



	m=[]
	count_part_number = 0
	m0 = [ t0,vertici_iniziali,data,count_part_number,neighbors ] 
	m.append(m0)
	
	box = []
	time = []
	
	father = []
	part_number = []
	
	vertici = []
	vertici.append(vertici_iniziali)
	
	
	
	
	vertici_per_plot=[]
	for i in range(len(vertici_iniziali)):
		vertici_per_plot.append(vertici_iniziali[i][0])
	box.append(vertici_per_plot)
	time.append(t0)
	father.append('nan')
	part_number.append(count_part_number)




	
	count=0
	for i in m:
		
		
		count += 1
		print('iterazione: ',count)

	
		try:

			 
			mondrian = MondrianPolygon_SingleCut(i[2],i[0],i[1],lifetime,i[3],dist_matrix,i[4])
			
			count1 = count_part_number+1
			mondrian[0][3][0] = count1+1
			m.append([mondrian[0][0],mondrian[0][1],mondrian[0][2],count_part_number+1,mondrian[0][3]])
			count_part_number += 1
			part_number.append(count_part_number)
			
			count2 = count_part_number+1
			mondrian[1][3][0] = count2-1
			m.append([mondrian[1][0],mondrian[1][1],mondrian[1][2],count_part_number+1,mondrian[1][3]])
			count_part_number += 1
			part_number.append(count_part_number)
			
			
			
			
			
			for j in range(2):
				vertici_per_plot=[]
				for h in range(len(mondrian[j][1])):
					vertici_per_plot.append(mondrian[j][1][h][0])
				vertici.append(mondrian[j][1])
				box.append(vertici_per_plot)
				time.append(mondrian[2])
				father.append(mondrian[3])
				

			
#part2 = [t0, l2, data2, neigh2]
#mondrian = [part1, part2, t0, father]			
			
			### calcolo partizioni vicine
			vicini_non_tagliati1 = mondrian[0][3][2:-1]
			for j in vicini_non_tagliati1:
				if type(j)!=int:
					continue
				for k in range(len(m[j][4])):
					if m[j][4][k] == i[3]:
						m[j][4][k] = count1
				
			
			vicini_non_tagliati2 = mondrian[1][3][2:-1]
			for j in vicini_non_tagliati2:
				if type(j)!=int:
					continue
				for k in range(len(m[j][4])):
					if m[j][4][k] == i[3]:
						m[j][4][k] = count2		
				
			
			l1 = mondrian[0][1]
			l2 = mondrian[1][1]
			#primo lato tagliato l1
			if type(mondrian[0][3][1])==int:
				segmento1 = [ l1[1][1],l1[1][0] ]
				segmento2 = [ l2[-1][1],l2[-1][0] ]
				neigh1 = count1
				neigh2 = count2
				for j in range(len(m[mondrian[0][3][1]][4])):
					if m[mondrian[0][3][1]][4][j]==i[3]:
						m[mondrian[0][3][1]][1][j] = segmento1
						m[mondrian[0][3][1]][1].insert(j+1,segmento2)
						m[mondrian[0][3][1]][4][j] = neigh1
						m[mondrian[0][3][1]][4].insert(j+1,neigh2)

				
			#secondo lato tagliato l1
			if type(mondrian[0][3][-1])==int:
				segmento1 = [ l2[1][1],l2[1][0] ]
				segmento2 = [ l1[-1][1],l1[-1][0] ]
				neigh1 = count2
				neigh2 = count1
				for j in range(len(m[mondrian[0][3][-1]][4])):
					if m[mondrian[0][3][-1]][4][j]==i[3]:
						m[mondrian[0][3][-1]][1][j] = segmento1
						m[mondrian[0][3][-1]][1].insert(j+1,segmento2)
						m[mondrian[0][3][-1]][4][j] = neigh1
						m[mondrian[0][3][-1]][4].insert(j+1,neigh2)						
				### fine calcolo partizioni vicine						

			
			# se voglio fermarmi al primo taglio
			#if len(m)==3:
			#	break
			

		except  TypeError:
			continue
		
	
	neigh_part = []	
	for i in m:
		#x = np.array(i[4])
		#neigh_part.append(x[~np.isnan(x)].astype(int))
		neigh_part.append(i[4])

	part = {'time':time,'father':father,'part_number':part_number,'neighbors':neigh_part,'box':box,'vertici':vertici}
	part = pd.DataFrame(part)
	

	

	
	leaf = []
	for i in range(len(part)):
		if part['part_number'].iloc[i] not in part['father'].unique():
			leaf.append(True)
		else:
			leaf.append(False)


		
	part['leaf'] = leaf

	part = part[['time', 'father', 'part_number', 'neighbors', 'leaf', 'vertici', 'box']]	
	
	
	
	return m,box,part
		
