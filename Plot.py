import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pyplot import cm





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
		


