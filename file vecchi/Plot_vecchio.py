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
def PlotPolygon(m,part):
	


	if isinstance(part, pd.DataFrame):
			

		#sns.set_style('whitegrid')
		fig,ax = plt.subplots()
		
	
		for i in range(len(part.query('leaf==True'))):
			box_new = part.query('leaf==True')['box'].iloc[i]
			p = Polygon(box_new, facecolor = 'none', edgecolor='b')
			ax.add_patch(p)
			
			b = pd.DataFrame(box_new)
			x_avg = np.mean(b[0])
			y_avg = np.mean(b[1])
			ax.text(x_avg,y_avg,part.query('leaf==True')['part_number'].iloc[i])
			
			if isinstance(m, list):
				data = pd.DataFrame(m[part.query('leaf==True')['part_number'].iloc[i]])
				ax.scatter(data['0'],data['1'],s=10,alpha=0.5,color='b')
		if isinstance(m, np.ndarray):
			X = m.copy()
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
				
				data = pd.DataFrame(m[part.query('leaf==True')['part_number'].iloc[i]])
				ax.scatter(data['0'],data['1'],s=10,alpha=0.5,color='b')
		#ax.scatter(X[:,0],X[:,1],s=10,alpha=0.5)

		#xmin = box_new[0][0][0]-0.05
		#ymin = box_new[0][0][1]-0.05
		#xmax = box_new[0][2][0]+0.05
		#ymax = box_new[0][2][1]+0.05
			
		#ax.set_xlim(xmin,xmax)
		#ax.set_ylim(ymin,ymax)
		
		
		plt.show()
		
	
	return










def PlotClass_2D(m,part,conn_comp,number_of_clusters,name_file):
	
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
			
			data = pd.DataFrame(m[int(list(conn_comp[number_of_clusters-1][j])[k])])
			ax.scatter(data['0'],data['1'],s=10,alpha=0.5,color='b')
	#ax.scatter(X[:,0],X[:,1],color='b',s=10,alpha=0.5)
	
	#xmin = part['box'].iloc[0][0][0]-0.05
	#ymin = part['box'].iloc[0][0][1]-0.05
	#xmax = part['box'].iloc[0][2][0]+0.05
	#ymax = part['box'].iloc[0][2][1]+0.05
	
	#ax.set_xlim(xmin,xmax)
	#ax.set_ylim(ymin,ymax)		
					
	plt.show()
	if name_file != False:
		plt.savefig(name_file)
		
	return









from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




def PlotClass_3D(m,part,conn_comp,number_of_clusters):


	p = part.query('leaf==True')
	p.index = np.arange(len(p))
	
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	#ax.scatter3D(X[:,0],X[:,1],X[:,2],alpha=0.5)
	color=cm.rainbow(np.linspace(0,1,len(conn_comp[number_of_clusters-1])))
	data = pd.DataFrame()
	for j in range(len(conn_comp[number_of_clusters-1])):
		for k in range(len(conn_comp[number_of_clusters-1][j])):
			p2 = p[p['part_number']==list(conn_comp[number_of_clusters-1][j])[k]].copy()
			p2.index = [0]
			verts = p2['box'][0]
			hull = ConvexHull(verts)
			faces = hull.simplices
			for s in faces:
				sq = [[verts[s[0]][0], verts[s[0]][1], verts[s[0]][2]],
				      [verts[s[1]][0], verts[s[1]][1], verts[s[1]][2]],
					  [verts[s[2]][0], verts[s[2]][1], verts[s[2]][2]]]
				f = Poly3DCollection([sq],linewidths=0.01)
				f.set_color(color[j])
				f.set_alpha(0.1)
				ax.add_collection3d(f)
			data_k = pd.DataFrame(m[list(conn_comp[number_of_clusters-1][j])[k]])
			data = pd.concat([data,data_k])
	data.index = np.arange(len(data))
	ax.scatter3D(data['0'],data['1'],data['2'],alpha=0.5,color='b')

					
	plt.show()
	
	
	
	#color=cm.rainbow(np.linspace(0,1,len(conn_comp[number_of_clusters-1])))
	for j in range(len(conn_comp[number_of_clusters-1])):
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		#ax.scatter3D(X[:,0],X[:,1],X[:,2],alpha=0.5)
		ax.scatter3D(data['0'],data['1'],data['2'],alpha=0.5,color='b')
		for k in range(len(conn_comp[number_of_clusters-1][j])):
			p2 = p[p['part_number']==list(conn_comp[number_of_clusters-1][j])[k]].copy()
			p2.index = [0]
			verts = p2['box'][0]
			hull = ConvexHull(verts)
			faces = hull.simplices
			for s in faces:
				sq = [[verts[s[0]][0], verts[s[0]][1], verts[s[0]][2]],
				      [verts[s[1]][0], verts[s[1]][1], verts[s[1]][2]],
					  [verts[s[2]][0], verts[s[2]][1], verts[s[2]][2]]]
				f = Poly3DCollection([sq],linewidths=0.01)
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



# nuovo metodo

def Plot2D(part,list_m,list_p,number_of_clusters,name_file):

	p = pd.DataFrame(list_p[number_of_clusters-1])
	
	#for i in range(len(p)):
	#	p['merged_part'].iloc[i].append(p['part_number'].iloc[i])

	#sns.set_style('whitegrid')
	fig,ax = plt.subplots()
		
	color=cm.rainbow(np.linspace(0,1,len(p)))
	for i in range(len(p)):
		box = part[part['part_number']==p['part_number'].iloc[i]]['box'][0]
		pol = Polygon(box, facecolor=color[i], alpha=0.3, edgecolor='black')
		ax.add_patch(pol)
			
		b = pd.DataFrame(box)
		x_avg = np.mean(b[0])
		y_avg = np.mean(b[1])
		ax.text(x_avg,y_avg,int(p['part_number'].iloc[i]))
		for j in p['merged_part'].iloc[i]:
			box = part[part['part_number']==j]['box'][0]
			pol = Polygon(box, facecolor=color[i], alpha=0.3, edgecolor='black')
			ax.add_patch(pol)
			
			b = pd.DataFrame(box)
			x_avg = np.mean(b[0])
			y_avg = np.mean(b[1])
			ax.text(x_avg,y_avg,int(j))
		
	data = pd.DataFrame(list_m[0][0])
	ax.scatter(data['0'],data['1'],s=10,alpha=0.5,color='b')
		
	plt.show()
	if name_file != False:
		plt.savefig(name_file)	
	return


	

def Plot3D(list_m_leaf,list_p,number_of_clusters):#part,

	p = pd.DataFrame(list_p[number_of_clusters-1])
	'''
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	color=cm.rainbow(np.linspace(0,1,len(p)))
	for i in range(len(p)):
		verts = part[part['part_number']==p['part_number'].iloc[i]]['box'][0]
		hull = ConvexHull(verts)
		faces = hull.simplices
		for s in faces:
			sq = [[verts[s[0]][0], verts[s[0]][1], verts[s[0]][2]],
				      [verts[s[1]][0], verts[s[1]][1], verts[s[1]][2]],
					  [verts[s[2]][0], verts[s[2]][1], verts[s[2]][2]]]
			f = Poly3DCollection([sq],linewidths=0.01)
			f.set_color(color[i])
			f.set_alpha(0.1)
			ax.add_collection3d(f)
		for j in p['merged_part'].iloc[i]:
			verts = part[part['part_number']==j]['box'][0]
			hull = ConvexHull(verts)
			faces = hull.simplices
			for s in faces:
				sq = [[verts[s[0]][0], verts[s[0]][1], verts[s[0]][2]],
				      [verts[s[1]][0], verts[s[1]][1], verts[s[1]][2]],
					  [verts[s[2]][0], verts[s[2]][1], verts[s[2]][2]]]
				f = Poly3DCollection([sq],linewidths=0.01)
				f.set_color(color[i])
				f.set_alpha(0.1)
				ax.add_collection3d(f)
				
		#data = pd.DataFrame(list_m_leaf[number_of_clusters-1][i])
		#ax.scatter(data['0'],data['1'],data['2'],s=10,alpha=0.7,color='b')
	ax.scatter3D(list(df.query('cl==0')['x']),list(df.query('cl==0')['y']),list(df.query('cl==0')['z']))
	ax.scatter3D(list(df.query('cl==1')['x']),list(df.query('cl==1')['y']),list(df.query('cl==1')['z']))
	ax.scatter3D(list(df.query('cl==2')['x']),list(df.query('cl==2')['y']),list(df.query('cl==2')['z']))

	#data = pd.DataFrame(list_m[0][0])
	#ax.scatter(data['0'],data['1'],data['2'],s=10,alpha=0.5,color='b')

	plt.show()
	
	'''
	color=cm.rainbow(np.linspace(0,1,len(p)))
	for i in range(len(p)):
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		verts = part[part['part_number']==p['part_number'].iloc[i]]['box'][0]
		hull = ConvexHull(verts)
		faces = hull.simplices
		for s in faces:
			sq = [[verts[s[0]][0], verts[s[0]][1], verts[s[0]][2]],
				      [verts[s[1]][0], verts[s[1]][1], verts[s[1]][2]],
					  [verts[s[2]][0], verts[s[2]][1], verts[s[2]][2]]]
			f = Poly3DCollection([sq],linewidths=0.01)
			f.set_color(color[i])
			f.set_alpha(0.1)
			ax.add_collection3d(f)
		for j in p['merged_part'].iloc[i]:
			verts = part[part['part_number']==j]['box'][0]
			hull = ConvexHull(verts)
			faces = hull.simplices
			for s in faces:
				sq = [[verts[s[0]][0], verts[s[0]][1], verts[s[0]][2]],
				      [verts[s[1]][0], verts[s[1]][1], verts[s[1]][2]],
					  [verts[s[2]][0], verts[s[2]][1], verts[s[2]][2]]]
				f = Poly3DCollection([sq],linewidths=0.01)
				f.set_color(color[i])
				f.set_alpha(0.1)
				ax.add_collection3d(f)
		for l in range(len(p)):
			data = pd.DataFrame(list_m_leaf[number_of_clusters-1][l])
			ax.scatter(data['0'],data['1'],data['2'],s=10,alpha=0.5,color='b')
#		ax.scatter3D(list(df.query('cl==0')['x']),list(df.query('cl==0')['y']),list(df.query('cl==0')['z']))
#		ax.scatter3D(list(df.query('cl==1')['x']),list(df.query('cl==1')['y']),list(df.query('cl==1')['z']))
#		ax.scatter3D(list(df.query('cl==2')['x']),list(df.query('cl==2')['y']),list(df.query('cl==2')['z']))
		plt.show()
	'''
	
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	color=cm.rainbow(np.linspace(0,1,len(p)))
	for i in range(len(p)):
		data = pd.DataFrame(list_m_leaf[number_of_clusters-1][i])
		ax.scatter(data['0'],data['1'],data['2'],s=10,alpha=0.7,color=color[i])
		
		plt.show()'''
	
	return

#%%
	color=cm.rainbow(np.linspace(0,1,len(p)))
	for i in range(len(p)):
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		Ai = part[part['part_number']==p['part_number'].iloc[i]]['polytope'][0]['A']
		bi = part[part['part_number']==p['part_number'].iloc[i]]['polytope'][0]['b']
		pi = pc.Polytope(np.array(Ai), np.array(bi))
		for j in p['merged_part'].iloc[i]:
			Aj = part[part['part_number']==j]['polytope'][0]['A']
			bj =  part[part['part_number']==j]['polytope'][0]['b']
			pj = pc.Polytope(np.array(Aj), np.array(bj))
			pi = pi.union(pj)
			verts = pypoman.compute_polytope_vertices(pi.A,pi.b)
		hull = ConvexHull(verts)
		faces = hull.simplices
		for s in faces:
			sq = [[verts[s[0]][0], verts[s[0]][1], verts[s[0]][2]],
				      [verts[s[1]][0], verts[s[1]][1], verts[s[1]][2]],
					  [verts[s[2]][0], verts[s[2]][1], verts[s[2]][2]]]
			f = Poly3DCollection([sq],linewidths=0.01)
			f.set_color(color[i])
			f.set_alpha(0.1)
			ax.add_collection3d(f)
		data = pd.DataFrame(list_m[0][0])
		ax.scatter(data['0'],data['1'],data['2'],s=10,alpha=0.5,color='b')

	

#%%

def Plot2D_binario(n,list_part,list_p_tot,number_of_clusters,name_file):

#%%	
	I=[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

	fig,ax = plt.subplots()
	n=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]	
	for l in n:#range(len(list_part)):#n
		if l!=50:
	
			
			part = list_part[l]
			list_p = list_p_tot[l]
			
			p = pd.DataFrame(list_p[number_of_clusters-1])
			alpha = 1/len(list_part)
			
			i=0#I[l] #0
			box = part[part['part_number']==p['part_number'].iloc[i]]['box'][0]
			pol = Polygon(box, facecolor='black', alpha=alpha, edgecolor=None,linewidth=0.00001)
			ax.add_patch(pol)
	
			for j in p['merged_part'].iloc[i]:
				box = part[part['part_number']==j]['box'][0]
				pol = Polygon(box, facecolor='black', alpha=alpha, edgecolor=None,linewidth=0.00001)
				ax.add_patch(pol)
					
			xmin = part['box'].iloc[0][0][0]-0.05
			ymin = part['box'].iloc[0][0][1]-0.05
			xmax = part['box'].iloc[0][2][0]+0.05
			ymax = part['box'].iloc[0][2][1]+0.05
		
			ax.set_xlim(xmin,xmax)
			ax.set_ylim(ymin,ymax)	

	data = pd.DataFrame(list_m[0][0])
	ax.scatter(data['0'],data['1'],s=10,alpha=0.5,color='b')
		
#%%		
	plt.show()
	if name_file != False:
		plt.savefig(name_file)	
	return

#%% plot tagli paralleli binario

df = {'x':X[:,0],'y':X[:,1],'cl':y}
df=pd.DataFrame(df)
x_max = df.query('cl==1')['x'].max()
x_min = df.query('cl==1')['x'].min()
y_max = df.query('cl==1')['y'].max()
y_min = df.query('cl==1')['y'].min()
	

fig,ax = plt.subplots()
alpha = 1/len(list_part)

for j in range(len(list_part)):
	part = list_part[j].query('leaf==True').copy()
	part.index = np.arange(len(part))
			
	for i in range(len(part)):
		
		box_new = part['box'].iloc[i]
		x_df_max = max(np.array(box_new)[:,0])
		x_df_min = min(np.array(box_new)[:,0])
		y_df_max = max(np.array(box_new)[:,1])
		y_df_min = min(np.array(box_new)[:,1])
				
		controllo = 0
		#for m,n in zip([x_df_max,x_df_min],[y_df_max,y_df_min]):
		#	if (m>x_min) and (m<x_max) and (n>y_min) and (n<y_max):
		#		controllo = 1
		for m in [x_df_max,x_df_min]:
			for n in [y_df_max,y_df_min]:
				if (m>x_min) and (m<x_max) and (n>y_min) and (n<y_max):
					controllo=1
		for m in [x_df_max,x_df_min]:
			if (m>x_min) and (m<x_max) and (y_df_min<y_min) and (y_df_max>y_max):
				controllo=1					
					
		for n in [y_df_max,y_df_min]:
			if (x_df_min<x_min) and (x_df_max>x_max) and (n>y_min) and (n<y_max):
				controllo=1	 
				
		if controllo==0:
			p = Polygon(box_new, facecolor='black', alpha=alpha, edgecolor=None,linewidth=0.00001)
			ax.add_patch(p)
		
#xmin = list_part[0]['box'].iloc[0][0][0]-0.05
#ymin = list_part[0]['box'].iloc[0][0][1]-0.05
#xmax = list_part[0]['box'].iloc[0][2][0]+0.05
#ymax = list_part[0]['box'].iloc[0][2][1]+0.05
		
#ax.set_xlim(xmin,xmax)
#ax.set_ylim(ymin,ymax)	

ax.scatter(X[:,0],X[:,1],color='b',s=10,alpha=0.5)
			
