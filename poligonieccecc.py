import numpy as np
#from numpy.random import choice
import pandas as pd	
#from scipy.spatial import distance

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
#from matplotlib.pyplot import cm





#%%


#part.query('leaf==True')[['time', 'father', 'part_number', 'neighbors', 'leaf']]
# coppie di vertici 
#vertici_iniziali=[ [[0,0],[1,0]], [[1,0],[1,1]], [[1,1],[0,1]], [[0,1],[0,0]]]
t0=0
lifetime=1.5
#dist_matrix = DistanceMatrix(X)
m,box,part=MondrianPolygon(X,t0,lifetime,dist_matrix)

PlotPolygon(X,part)


#%%


part = []
for i in range(100):
	print('AAAAAAAAAAAAAAAAAAA',i)
	m,box,part_i=MondrianPolygon(X,t0,lifetime,dist_matrix)
	part.append(part_i)
	
PlotPolygon(X,part)



#%% per salvare su file

for i in range(len(part)):
	part[i].to_csv('98iterazioniNoheader.txt',mode='a',index=False,sep='\t',header=False)


#%%

sns.set_style('whitegrid')
fig,ax = plt.subplots()

#color=cm.rainbow(np.linspace(0,1,len(box)))
#for i,c in zip(range(len(box)),color):
for i in range(len(box)):
	p = Polygon(box[i], facecolor = 'none', edgecolor='b')
	ax.add_patch(p)
	
	
	ax.scatter(m[i][2][0],m[i][2][1],color='b')
	
xmin = box[0][0][0]-0.05
ymin = box[0][0][1]-0.05
xmax = box[0][2][0]+0.05
ymax = box[0][2][1]+0.05
	
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)

#dati = pd.DataFrame(X)
#ax.scatter(dati[0],dati[1])

plt.show()


#%% uso solo foglie    metto numeri partizioni

sns.set_style('whitegrid')
fig,ax = plt.subplots()

#color=cm.rainbow(np.linspace(0,1,len(box)))
#for i,c in zip(range(len(box)),color):
for i in range(len(part.query('leaf==True'))):
	box_new = part.query('leaf==True')['box'].iloc[i]
	p = Polygon(box_new, facecolor = 'none', edgecolor='b')
	ax.add_patch(p)
	
	b = pd.DataFrame(box_new)
	x_avg = np.mean(b[0])
	y_avg = np.mean(b[1])
	ax.text(x_avg,y_avg,part.query('leaf==True')['part_number'].iloc[i])
	
	
	#ax.scatter(m[i][2][0],m[i][2][1],color='b')
	
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
	
	p = Polygon(box[i], facecolor = 'none',edgecolor='b')
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

