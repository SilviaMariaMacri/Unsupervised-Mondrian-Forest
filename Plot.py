import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pypoman
import math
import pandas as pd




def compute_vertices(poly):
	
	A = poly.A
	b = poly.b
	vert = pypoman.compute_polytope_vertices(A,b)
	# ordino vertici:
	# compute centroid
	cent=(sum([v[0] for v in vert])/len(vert),sum([v[1] for v in vert])/len(vert))
	# sort by polar angle
	vert.sort(key=lambda v: math.atan2(v[1]-cent[1],v[0]-cent[0]))
	
	return vert			





def plot2D_partitioning(data,part_space):
	
	p = part_space.query('leaf==True').copy()
	
	fig,ax = plt.subplots()
	for i in range(len(p)):
		poly = p['polytope'].iloc[i]
		vert = compute_vertices(poly)
		poly_for_plot = Polygon(vert, facecolor = 'none', edgecolor='b')
		ax.add_patch(poly_for_plot)
			
		x_avg = np.mean(np.array(vert)[:,0])
		y_avg = np.mean(np.array(vert)[:,1])
		ax.text(x_avg,y_avg,p['id_number'].iloc[i])
			
	ax.scatter(data[:,0],data[:,1],s=10,alpha=0.5,color='b')
	
	plt.show()
	
	return






def plot2D_merging(data,part_space,merg_space,number_of_clusters):

	p = merg_space[number_of_clusters-1].copy()
	
	color = cm.rainbow(np.linspace(0,1,len(p)))
	fig,ax = plt.subplots()
	for i in range(len(p)):
		poly_number = p['id_number'].iloc[i]
		poly = list(part_space[part_space['id_number']==poly_number]['polytope'])[0]
		vert = compute_vertices(poly)
		poly_for_plot = Polygon(vert, facecolor=color[i], alpha=0.3, edgecolor='black')
		ax.add_patch(poly_for_plot)
		
		x_avg = np.mean(np.array(vert)[:,0])
		y_avg = np.mean(np.array(vert)[:,1])
		ax.text(x_avg,y_avg,poly_number)

		for j in p['merged'].iloc[i]:
			poly = list(part_space[part_space['id_number']==j]['polytope'])[0]
			vert = compute_vertices(poly)
			poly_for_plot = Polygon(vert, facecolor=color[i], alpha=0.3, edgecolor='black')
			ax.add_patch(poly_for_plot)

			x_avg = np.mean(np.array(vert)[:,0])
			y_avg = np.mean(np.array(vert)[:,1])
			ax.text(x_avg,y_avg,j)

	#data = m[0].copy()
	ax.scatter(data[:,0],data[:,1],s=10,alpha=0.5,color='b')
		
	return




def plot3D(data,part_space,merg_space,merg_data,number_of_clusters,plot_data,plot_space):
	
	p = merg_space[number_of_clusters-1].copy()
	cl_data = merg_data[number_of_clusters-1]
	data = data[:,0:-1]
	data = pd.DataFrame(data)
	
	if plot_data == True:
		
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		color=cm.rainbow(np.linspace(0,1,len(p)))
		for i in range(len(p)):
			data['cl'] = cl_data
			data_i = data.query('cl=='+str(i)).copy()
			ax.scatter(data_i[0],data_i[1],data_i[2],s=10,alpha=0.7,color=color[i])
		plt.show()	


	if plot_space == True:
		
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		color=cm.rainbow(np.linspace(0,1,len(p)))
		for i in range(len(p)):
			part_number = p['id_number'].iloc[i]
			poly = list(part_space[part_space['id_number']==part_number]['polytope'])[0]
			verts = compute_vertices(poly)
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
			
			for j in p['merged'].iloc[i]:
				poly = list(part_space[part_space['id_number']==j]['polytope'])[0]
				verts = compute_vertices(poly)
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
					
		ax.scatter(data[0],data[1],data[2],s=10,alpha=0.7,color='b')
		
		plt.show()
		
		
		color=cm.rainbow(np.linspace(0,1,len(p)))
		for i in range(len(p)):
			fig = plt.figure()
			ax = plt.axes(projection='3d')
			part_number = p['id_number'].iloc[i]
			poly = list(part_space[part_space['id_number']==part_number]['polytope'])[0]
			verts = compute_vertices(poly)
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
			for j in p['merged'].iloc[i]:
				poly = list(part_space[part_space['id_number']==j]['polytope'])[0]
				verts = compute_vertices(poly)
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
					
			ax.scatter(data[0],data[1],data[2],s=10,alpha=0.7,color='b')
		
		plt.show()

	
	return



def plot_AMI(ami_mean,ami_std):
	
	fig,ax = plt.subplots()
	ax.plot(np.arange(2,len(ami_mean)+1),ami_mean[1:],linewidth=0.7)
	ax.scatter(np.arange(2,len(ami_mean)+1),ami_mean[1:],s=10)
	ax.fill_between(np.arange(2,len(ami_mean)+1), ami_mean[1:]-np.array(ami_std[1:])/2, ami_mean[1:]+np.array(ami_std[1:])/2,alpha=0.2,color='b')
	ax.set_xlabel('Number of Clusters')
	ax.set_ylabel('Adjusted Mutual Information')
	plt.show()
	
	return 




'''
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
'''