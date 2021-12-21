import numpy as np
#from numpy.random import choice
import pandas as pd	
#from scipy.spatial import distance

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
#from matplotlib.pyplot import cm


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

	

