import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection





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




def Plot3D(list_m_leaf,list_p,part,number_of_clusters):
	

	p = pd.DataFrame(list_p[number_of_clusters-1])
	
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
				
		data = pd.DataFrame(list_m_leaf[number_of_clusters-1][i])
		ax.scatter(data['0'],data['1'],data['2'],s=10,alpha=0.7,color='b')
	'''
	ax.scatter3D(list(df.query('cl==0')['x']),list(df.query('cl==0')['y']),list(df.query('cl==0')['z']))
	ax.scatter3D(list(df.query('cl==1')['x']),list(df.query('cl==1')['y']),list(df.query('cl==1')['z']))
	ax.scatter3D(list(df.query('cl==2')['x']),list(df.query('cl==2')['y']),list(df.query('cl==2')['z']))
	'''

	#data = pd.DataFrame(list_m[0][0])
	#ax.scatter(data['0'],data['1'],data['2'],s=10,alpha=0.5,color='b')

	plt.show()
	
	
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
	
	
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	color=cm.rainbow(np.linspace(0,1,len(p)))
	for i in range(len(p)):
		data = pd.DataFrame(list_m_leaf[number_of_clusters-1][i])
		ax.scatter(data['0'],data['1'],data['2'],s=10,alpha=0.7,color=color[i])
		
		plt.show()
	
	return





def Plot2D_binario(n,list_part,list_p_tot,number_of_clusters,name_file,list_m):

	
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
		
	
	plt.show()
	if name_file != False:
		plt.savefig(name_file)	
	return

