import numpy as np
from numpy.random import choice
import pandas as pd	
from scipy.spatial import distance

#import matplotlib.pyplot as plt
#from matplotlib.patches import Polygon









def	Cut_without_data(l,ld):

	
	n_vert = np.arange(len(l))
		
	p=ld/sum(np.array(ld))
	lati = choice(n_vert,p=p,replace=False,size=2)
	lati.sort()
		
	point1 = np.array(l[lati[0]][0]) + (np.array(l[lati[0]][1]) - np.array(l[lati[0]][0]))*np.random.uniform(0,1)
	point2 = np.array(l[lati[1]][0]) + (np.array(l[lati[1]][1]) - np.array(l[lati[1]][0]))*np.random.uniform(0,1)

	return point1,point2,lati


 



# vale solo in 2D
def Cut_with_data(data,l):
	
	
	n_vert = np.arange(len(l))
	
	dist_matrix = DistanceMatrix(data)
	
	# a*x + b*y = c      retta cut
	matrix = dist_matrix[dist_matrix['dist']==dist_matrix['dist'].max()].copy()
	matrix.index = np.arange(len(matrix))
	a = matrix['norm_vect_0'].iloc[0] #versore0
	b = matrix['norm_vect_1'].iloc[0] #versore1
	c = matrix['magnitude_norm_vect'].iloc[0] # -modulo   
	
	
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
	
	cut = [matrix['norm_vect_0'].iloc[0], matrix['norm_vect_1'].iloc[0], matrix['magnitude_norm_vect'].iloc[0]] 
	
	matrix = matrix[['dim0_point','dim1_point', 'distance_point_cut']]
	
		
	return points,matrix,cut














def MondrianPolygon_SingleCut(data,t0,l,lifetime,father):
	

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
	
	
	if len(data) <= 2: #dall'altra parte c'è scritto <= 3 per storia del calcolo varianza
		return

	

	#senza dati		
	#point1,point2,lati = Cut_without_data(l,ld)
	#con dati
	points,matrix,cut = Cut_with_data(data,l)
	print('cut: ',cut)
	print('intersezioni: ',points)
	print(matrix)



#if (cut[0]>0 and cut[1]>0 and cut[2]>0) or (cut[0]<0 and cut[1]<0 and cut[2]<0):
#	print('1') 
#if (cut[0]>0 and cut[1]<0 and cut[2]<0) or (cut[0]<0 and cut[1]>0 and cut[2]>0):
#	print('2')
#if (cut[0]>0 and cut[1]>0 and cut[2]<0) or (cut[0]<0 and cut[1]<0 and cut[2]>0):
#	print('3') 
#if (cut[0]>0 and cut[1]<0 and cut[2]>0) or (cut[0]<0 and cut[1]>0 and cut[2]<0):
#	print('4')
			
	
	
	
	# quadrante 1 e 2
	if (cut[0]>0 and cut[1]>0 and cut[2]>0) or (cut[0]<0 and cut[1]<0 and cut[2]<0) or (cut[0]>0 and cut[1]<0 and cut[2]<0) or (cut[0]<0 and cut[1]>0 and cut[2]>0):
		if points['x'].iloc[0] > points['x'].iloc[1]:
			print('1')
			data1 = matrix.query('distance_point_cut<0')[['dim0_point','dim1_point']].copy()
			data2 = matrix.query('distance_point_cut>0')[['dim0_point','dim1_point']].copy()
		else:
			print('2')
			data2 = matrix.query('distance_point_cut<0')[['dim0_point','dim1_point']].copy()
			data1 = matrix.query('distance_point_cut>0')[['dim0_point','dim1_point']].copy()
		
		
		#xmax,xmin in ordine:
		#	l1 = dist - 
		#	l2 = dist +
		#xmin,xmax:
		#	l1 = dist +
		#	l2 = dist -
		
		
		
	# quadrante 3 e 4
	#if (cut[0]>0 and cut[1]>0 and cut[2]<0) or (cut[0]<0 and cut[1]<0 and cut[2]>0) or (cut[0]>0 and cut[1]<0 and cut[2]>0) or (cut[0]<0 and cut[1]>0 and cut[2]<0):
	else:
		if points['x'].iloc[0] > points['x'].iloc[1]:
			print('3')
			data1 = matrix.query('distance_point_cut>0')[['dim0_point','dim1_point']].copy()
			data2 = matrix.query('distance_point_cut<0')[['dim0_point','dim1_point']].copy()
		else:
			print('4')
			data1 = matrix.query('distance_point_cut<0')[['dim0_point','dim1_point']].copy()
			data2 = matrix.query('distance_point_cut>0')[['dim0_point','dim1_point']].copy()



		#xmax,xmin in        ordine:
		#	l1 = dist + 
		#	l2 = dist -
		#xmin,xmax:
		#	l1 = dist -
		#	l2 = dist +
		
	data1.index = np.arange(len(data1))
	data1.columns = [0,1]
	data2.index = np.arange(len(data2))
	data2.columns = [0,1]
	
	point1 = list(points[['x','y']].iloc[0])
	point2 = list(points[['x','y']].iloc[1])
	lati = list(points['lati'])
	
###################### fine parte con dati	


	
	l1 = l.copy()
	l2 = l.copy()		
	
		
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
	
	
	
	risultato1 = [t0, l1, data2]
	risultato2 = [t0, l2, data1]
	risultato = [risultato1, risultato2, t0, father]
	
	
	
	return risultato










def MondrianPolygon(X,t0,lifetime): 
	
	
	
	data = pd.DataFrame(X)
	
	# 2D
	vertici_iniziali = []
	length0 = data[0].max() - data[0].min()
	length1 = data[1].max() - data[1].min()
	vertici_iniziali.append([[data[0].min() - length0*0.05,data[1].min() - length1*0.05],[data[0].max() + length0*0.05,data[1].min() - length1*0.05]])
	vertici_iniziali.append([[data[0].max() + length0*0.05,data[1].min() - length1*0.05],[data[0].max() + length0*0.05,data[1].max() + length1*0.05]])
	vertici_iniziali.append([[data[0].max() + length0*0.05,data[1].max() + length1*0.05],[data[0].min() - length0*0.05,data[1].max() + length1*0.05]])
	vertici_iniziali.append([[data[0].min() - length0*0.05,data[1].max() + length1*0.05],[data[0].min() - length0*0.05,data[1].min() - length1*0.05]])

	

	m=[]
	count_part_number = 0
	m0 = [ t0,vertici_iniziali,data,count_part_number ] 
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
			


		 
			mondrian = MondrianPolygon_SingleCut(i[2],i[0],i[1],lifetime,i[3])
			
			m.append([mondrian[0][0],mondrian[0][1],mondrian[0][2],count_part_number+1])
			count_part_number += 1
			part_number.append(count_part_number)
			
			m.append([mondrian[1][0],mondrian[1][1],mondrian[1][2],count_part_number+1])
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
		




