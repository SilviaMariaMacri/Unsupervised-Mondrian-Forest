import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Polygon
import seaborn as sns





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
				
				
	ax.scatter(data[data['class']==0][0],data[data['class']==0][1],data[data['class']==0][2],alpha=0.7)
	ax.scatter(data[data['class']==1][0],data[data['class']==1][1],data[data['class']==1][2],alpha=0.7)
	
				
	plt.show()
	
	return

	
	
	
	
def PartitionPlot2D(X,y,part):
	
	
	data = pd.DataFrame(X)
	data['class'] = y


	if isinstance(part, pd.DataFrame):
		
		p = part[part['leaf']==True]	
		
		
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
		
		for i in range(len(p)):
			
			ax.vlines(p['min0'].iloc[i],p['min1'].iloc[i],p['max1'].iloc[i])		
			ax.vlines(p['max0'].iloc[i],p['min1'].iloc[i],p['max1'].iloc[i])
			ax.hlines(p['min1'].iloc[i],p['min0'].iloc[i],p['max0'].iloc[i])
			ax.hlines(p['max1'].iloc[i],p['min0'].iloc[i],p['max0'].iloc[i])
			ax.text(p['min0'].iloc[i],p['min1'].iloc[i],p['part_number'].iloc[i])
	
		#ax.scatter(data[data['class']==0][0],data[data['class']==0][1],alpha=0.7)
		#ax.scatter(data[data['class']==1][0],data[data['class']==1][1],alpha=0.7)
		ax.scatter(X[:,0],X[:,1])
		plt.show()
		
	
	if isinstance(part, list):
		
		
		color=cm.rainbow(np.linspace(0,1,len(part)))
		
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
		
		for j,c in zip(part,color):
			
			p = j[j['leaf']==True]	
		
			for i in range(len(p)):
				
				ax.vlines(p['min0'].iloc[i],p['min1'].iloc[i],p['max1'].iloc[i],alpha=0.3,color=c)		
				ax.vlines(p['max0'].iloc[i],p['min1'].iloc[i],p['max1'].iloc[i],alpha=0.3,color=c)
				ax.hlines(p['min1'].iloc[i],p['min0'].iloc[i],p['max0'].iloc[i],alpha=0.3,color=c)
				ax.hlines(p['max1'].iloc[i],p['min0'].iloc[i],p['max0'].iloc[i],alpha=0.3,color=c)
				#ax.text(p['min0'].iloc[i],p['min1'].iloc[i],p['part_number'].iloc[i])
		
		#ax.scatter(data[data['class']==0][0],data[data['class']==0][1],alpha=0.7)
		#ax.scatter(data[data['class']==1][0],data[data['class']==1][1],alpha=0.7)
		ax.scatter(X[:,0],X[:,1])	
		plt.show()		
			
	
	
	return
'''	
		number_of_clusters = 2
		p = part[part['leaf']==True]	
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
		
		for j in range(len(conn_comp_fin[number_of_clusters-1])):
			
			
			ax.vlines(p['min0'].iloc[i],p['min1'].iloc[i],p['max1'].iloc[i])		
			ax.vlines(p['max0'].iloc[i],p['min1'].iloc[i],p['max1'].iloc[i])
			ax.hlines(p['min1'].iloc[i],p['min0'].iloc[i],p['max0'].iloc[i])
			ax.hlines(p['max1'].iloc[i],p['min0'].iloc[i],p['max0'].iloc[i])
			ax.text(p['min0'].iloc[i],p['min1'].iloc[i],p['part_number'].iloc[i])
	
		ax.scatter(X[:,0],X[:,1],alpha=0.7)
		plt.show()
'''


def PartitionPlot(X,y,part):
	
	if len(X[0]) == 2:
		PartitionPlot2D(X,y,part)
		
	if len(X[0]) == 3:
		PartitionPlot3D(X,y,part)
		
		
	return
		





# funziona anche con tagli paralleli
def PlotPolygon(X,part):
	


	if isinstance(part, pd.DataFrame):
			

		sns.set_style('whitegrid')
		fig,ax = plt.subplots()
		
	
		for i in range(len(part.query('leaf==True'))):
			box_new = part.query('leaf==True')['box'].iloc[i]
			p = Polygon(box_new, facecolor = 'none', edgecolor='b')
			ax.add_patch(p)
			
			b = pd.DataFrame(box_new)
			x_avg = np.mean(b[0])
			y_avg = np.mean(b[1])
			ax.text(x_avg,y_avg,part.query('leaf==True')['part_number'].iloc[i])
			
			
		ax.scatter(X[:,0],X[:,1],color='b',s=10,alpha=0.5)
			
		#xmin = box_new[0][0][0]-0.05
		#ymin = box_new[0][0][1]-0.05
		#xmax = box_new[0][2][0]+0.05
		#ymax = box_new[0][2][1]+0.05
			
		#ax.set_xlim(xmin,xmax)
		#ax.set_ylim(ymin,ymax)
		
		
		plt.show()
		
		
		
	if isinstance(part, list):
		
		
		color=cm.rainbow(np.linspace(0,1,len(part)))
		
		fig, ax = plt.subplots()
		
		for j,c in zip(part,color):
			
			for i in range(len(j.query('leaf==True'))):
				box_new = j.query('leaf==True')['box'].iloc[i]
				p = Polygon(box_new, facecolor = 'none', edgecolor='b',alpha=0.05)
				ax.add_patch(p)
				
			
		ax.scatter(X[:,0],X[:,1],s=10,alpha=0.5)
			
		#xmin = box_new[0][0][0]-0.05
		#ymin = box_new[0][0][1]-0.05
		#xmax = box_new[0][2][0]+0.05
		#ymax = box_new[0][2][1]+0.05
			
		#ax.set_xlim(xmin,xmax)
		#ax.set_ylim(ymin,ymax)
		
		
		plt.show()
		
	
	return










def PlotClass_2D(X,part,conn_comp,number_of_clusters):
	
	p = part.query('leaf==True').copy()
	p.index = np.arange(len(p))
		
	fig,ax = plt.subplots()
		 			
	color=cm.rainbow(np.linspace(0,1,len(conn_comp[number_of_clusters-1])))
	for j in range(len(conn_comp[number_of_clusters-1])):
		for k in range(len(conn_comp[number_of_clusters-1][j])):
			p2 = p[p['part_number']==list(conn_comp[number_of_clusters-1][j])[k]].copy()
			box = p2['box'].iloc[0].copy()
			poligono = Polygon(box, facecolor=color[j], alpha=0.5, edgecolor='black')
			ax.add_patch(poligono)
					
			b = pd.DataFrame(box)
			x_avg = np.mean(b[0])
			y_avg = np.mean(b[1])
			ax.text(x_avg,y_avg,p2['part_number'].iloc[0])
				
				
	ax.scatter(X[:,0],X[:,1],color='b',s=10,alpha=0.5)
	
	xmin = part['box'].iloc[0][0][0]-0.05
	ymin = part['box'].iloc[0][0][1]-0.05
	xmax = part['box'].iloc[0][2][0]+0.05
	ymax = part['box'].iloc[0][2][1]+0.05
	
	ax.set_xlim(xmin,xmax)
	ax.set_ylim(ymin,ymax)		
					
	plt.show()
	#plt.savefig(str(i))
		
	return









import mpl_toolkits.mplot3d as a3
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




def PlotClass_3D(X,part,conn_comp,number_of_clusters):


	p = part.query('leaf==True')
	p.index = np.arange(len(p))
	
	ax = plt.axes(projection='3d')
	ax.scatter3D(X[:,0],X[:,1],X[:,2],alpha=0.5)
	
	color=cm.rainbow(np.linspace(0,1,len(conn_comp[number_of_clusters-1])))
	for j in range(len(conn_comp[number_of_clusters-1])):
		for k in range(len(conn_comp[number_of_clusters-1][j])):
			p2 = p[p['part_number']==list(conn_comp[number_of_clusters-1][j])[k]].copy()
			p2.index = [0]
			verts = p2['box'][0]
			hull = ConvexHull(verts)
			faces = hull.simplices
			poly = []
			for s in faces:
				sq = [[verts[s[0]][0], verts[s[0]][1], verts[s[0]][2]],
				      [verts[s[1]][0], verts[s[1]][1], verts[s[1]][2]],
					  [verts[s[2]][0], verts[s[2]][1], verts[s[2]][2]]]
				f = a3.art3d.Poly3DCollection([sq],linewidths=0.01)
				f.set_color(color[j])
				f.set_alpha(0.1)
				ax.add_collection3d(f)	

					
	plt.show()
	return



#p1 = pc.Polytope()
#for k in range(len(conn_comp[number_of_clusters-1][j])):
#	p2 = p[p['part_number']==list(conn_comp[number_of_clusters-1][j])[k]].copy()
#	pp = pc.Polytope(np.array(p2['polytope'].iloc[i]['A']),np.array(p2['polytope'].iloc[i]['b']))					
#	p1 = pc.union(p1,pp)			
			#poly.append(pyny.Polygon(np.array(sq)))
		
#polyhedron = pyny.Polyhedron(poly)#.get_plotable3d(alpha=0.3)
#polyhedron.plot(ax=ax,opacity=0.5)



#get_area()[source]
#ax.add_collection3d(Poly3DCollection(verts, 
 #facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

#import pyny3d.geoms as pyny

#poly1 = pyny.Polygon(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]))
#poly2 = pyny.Polygon(np.array([[0, 0, 3], [0.5, 0, 3], [0.5, 0.5, 3], [0, 0.5, 3]]))
#polyhedron = pyny.Polyhedron.by_two_polygons(poly1, poly2)
#polyhedron.plot('b')

	














#makecircles = ok in automatico
#makemoons =  invertire solo ultimo
#dati1 = cambiare i=0,1,9,11


def PlotClass_binario(list_part_tot,list_conn_comp,number_of_clusters):

	fig,ax = plt.subplots()
	for i in range(len(list_part_tot)):
		
		
		part = pd.DataFrame(list_part_tot[i]).copy()
		connected_components = list_conn_comp[i].copy()
		
		p = part.query('leaf==True').copy()
		p.index = np.arange(len(p))
		
		
		j=2
		#if (i==1):
		#	j=1
		for k in range(len(connected_components[number_of_clusters-1][j])):
			p2 = p[p['part_number']==list(connected_components[number_of_clusters-1][j])[k]].copy()
			
			box = p2['box'].iloc[0].copy()
			poligono = Polygon(box, facecolor='black', alpha=1/16, edgecolor=None,linewidth=0.00001)
			ax.add_patch(poligono)
			

			

		xmin = part['box'].iloc[0][0][0]-0.05
		ymin = part['box'].iloc[0][0][1]-0.05
		xmax = part['box'].iloc[0][2][0]+0.05
		ymax = part['box'].iloc[0][2][1]+0.05
	
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)		
		
	plt.savefig('imm_binaria_'+str(i))
		

	return




#import cv2

	