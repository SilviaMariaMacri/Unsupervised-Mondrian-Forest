####### creazione file

import Mondrian
from sklearn import datasets

#import sys
#dati_prova,lifetime,exp,metric,number_of_iterations= sys.argv

lifetime = 5
exp = 5
number_of_iterations = 1
metric = 'min_corr'

name = 'dati_prova_lambda'+str(lifetime)+'_exp'+str(exp)+'_'
dat = datasets.make_blobs(n_samples=20,n_features=2,cluster_std=[1.0, 2.5, 0.5],random_state=10)#random_state=150
X = dat[0]
y = dat[1]
t0 = 0


Mondrian.MondrianTree(name,
				  X,
				  t0,
				  int(lifetime),
				  int(exp),
				  metric,
				  int(number_of_iterations)
				  )


######

import json
import pandas as pd
import pylab as plt
import numpy as np

import TreeComparison
import Plot

'####### leggo file .json'

#name
#number_of_iterations

list_part = []
list_m = []

list_p_tot = []
list_m_leaf_tot = []


for i in range(number_of_iterations):

	namefile = name+metric+'_'+str(i+1)

	part = json.load(open(namefile+'_part.json','r'))
	part = pd.DataFrame(part)
	m = json.load(open(namefile+'_m.json','r'))
	list_part.append(part)
	list_m.append(m)
	
	list_p = json.load(open(namefile+'_p.json','r'))
	list_m_leaf = json.load(open(namefile+'_m_leaf.json','r'))
	list_p_tot.append(list_p)
	list_m_leaf_tot.append(list_m_leaf)	
		
	
	
	
	
	
	
'####### AMI e FMI	'

class_data_tot = []
for i in range(len(list_m_leaf_tot)):
	list_m_leaf = list_m_leaf_tot[i].copy()
	classified_data = TreeComparison.AssignClass(list_m_leaf)
	class_data_tot.append(classified_data)
	
c_mean,c_std,c_tot = TreeComparison.AMI(class_data_tot,False)

	
number_of_clusters = 3
FMS = TreeComparison.FMI(class_data_tot,y,number_of_clusters)
FMS_medio = np.mean(FMS)
FMS_std = np.std(FMS)
	 
	
	

#color = ['d','b','orange','g','r','purple','cyan']
#label = []

fig,ax = plt.subplots()
#for i in range(len(label)):
ax.plot(np.arange(2,len(c_mean)+1),c_mean[1:],linewidth=0.7)
ax.scatter(np.arange(2,len(c_mean)+1),c_mean[1:],s=10)
ax.fill_between(np.arange(2,len(c_mean+1), c_mean[1:]-c_std[1:]/2, c_mean[1:]+c_std[1:]/2,alpha=0.2))
#ax.plot(np.arange(2,len(c_mean[i])+1),c_mean[i][1:],label=label[i],color=color[i],linewidth=0.7)
#ax.scatter(np.arange(2,len(c_mean[i])+1),c_mean[i][1:],s=10,color=color[i])
#ax.fill_between(np.arange(2,len(c_mean[i])+1), c_mean[i][1:]-c_std[i][1:]/2, c_mean[i][1:]+c_std[i][1:]/2,alpha=0.2,color=color[i])
ax.legend()#title='Exp'		
#plt.savefig('moons2D_coeff_exp'+label[i])
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Adjusted Mutual Information')
	





'####### confronto altri metodi di clustering'

from sklearn.cluster import KMeans,DBSCAN,SpectralClustering#,AgglomerativeClustering,Birch,mean_shift
from sklearn.metrics.cluster import fowlkes_mallows_score

n_clusters = 3
title = ['KMeans','DBSCAN','SpectralClustering']#,'MeanShift']'AgglomerativeClustering','Birch',
labels = []

FMS_kmeans=[]
for i in range(20):
	random_state=i
	kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)#, random_state=0
	FMS_kmeans.append(fowlkes_mallows_score(y,kmeans.labels_))
labels.append(kmeans.labels_)
	#print('kmeans: ',fowlkes_mallows_score(y,kmeans.labels_))


#AC = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
#labels.append(AC.labels_)
#print('AgglomerativeClustering: ',fowlkes_mallows_score(y,AC.labels_))

#B = Birch(n_clusters=n_clusters).fit(X)
#labels.append(B.labels_)
#print('Birch: ',fowlkes_mallows_score(y,B.labels_))

FMS_dbscan = []
eps= 0.15              #sfera o cilindro 0.049#0.27
for i in range(20):
	DBS = DBSCAN(eps=eps).fit(X)
	FMS_dbscan.append(fowlkes_mallows_score(y,DBS.labels_))
labels.append(DBS.labels_)
	#print('DBSCAN: ',fowlkes_mallows_score(y,DBS.labels_))
#'''
FMS_spectral = []
for i in range(20):
	random_state=i
	SC = SpectralClustering(n_clusters=n_clusters,affinity='nearest_neighbors',random_state=random_state).fit(X)#,random_state=0,assign_labels='discretize'
	FMS_spectral.append(fowlkes_mallows_score(y,SC.labels_))
	labels.append(SC.labels_)
	#print('SpectralClustering: ',fowlkes_mallows_score(y,SC.labels_))

#MS = mean_shift(X)
#labels.append(MS[1])
#print('MeanShift: ',fowlkes_mallows_score(y,MS[1]))


print('kmeans: ')
print(FMS_kmeans)
print('DBSCAN: ')
print(FMS_dbscan)
print('Spectral clustering: ')
print(FMS_spectral)


#2D
for i in range(len(labels)):
	#i=0
	data = {'X0':X[:,0],'X1':X[:,1],'y':list(labels[i])}
	
	data=pd.DataFrame(data)
	
	fig,ax = plt.subplots()
	ax.scatter(data.query('y==0')['X0'],data.query('y==0')['X1'],alpha=0.5,color='b')
	ax.scatter(data.query('y==1')['X0'],data.query('y==1')['X1'],alpha=0.5,color='orange')
	ax.scatter(data.query('y==2')['X0'],data.query('y==2')['X1'],alpha=0.5,color='g')
	ax.scatter(data.query('y==-1')['X0'],data.query('y==-1')['X1'],alpha=0.5,color='r')
	
	ax.scatter(data.query('y==3')['X0'],data.query('y==3')['X1'],alpha=0.5,color='purple')
	ax.scatter(data.query('y==4')['X0'],data.query('y==4')['X1'],alpha=0.5,color='cyan')
	#ax.set_title(title[i]+' eps='+str(eps))


#3D
for i in range(len(labels)):
	data = {'X0':X[:,0],'X1':X[:,1],'X2':X[:,2],'cl':list(labels[i])}
	
	data=pd.DataFrame(data)
	
	#cl = data['y'].unique()
	
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	#ax.scatter3D(np.array(data[data['y']==cl[0]])[:,0],np.array(data[data['y']==cl[0]])[:,1],np.array(data[data['y']==cl[0]])[:,2],alpha=0.5,color='b')
	#ax.scatter3D(np.array(data[data['y']==cl[1]])[:,0],np.array(data[data['y']==cl[1]])[:,1],np.array(data[data['y']==cl[1]])[:,2],alpha=0.5,color='orange')
	#ax.scatter3D(np.array(data[data['y']==cl[2]])[:,0],np.array(data[data['y']==cl[2]])[:,1],np.array(data[data['y']==cl[2]])[:,2],alpha=0.5,color='g')
	ax.scatter3D(np.array(data.query('cl==0'))[:,0],np.array(data.query('cl==0'))[:,1],np.array(data.query('cl==0'))[:,2],alpha=0.5,color='b')
	ax.scatter3D(np.array(data.query('cl==1'))[:,0],np.array(data.query('cl==1'))[:,1],np.array(data.query('cl==1'))[:,2],alpha=0.5,color='orange')
	ax.scatter3D(np.array(data.query('cl==2'))[:,0],np.array(data.query('cl==2'))[:,1],np.array(data.query('cl==2'))[:,2],alpha=0.5,color='g')
	ax.scatter3D(np.array(data.query('cl==-1'))[:,0],np.array(data.query('cl==-1'))[:,1],np.array(data.query('cl==-1'))[:,2],alpha=0.5,color='r')
	
	#ax.set_title(title[i])
	
	
	
	

'####### grafici'

#name = 'makeblobs_3D_'
#number_of_iterations = 10
for i in range(number_of_iterations):
	print(i)
	
	#i=2
	part = list_part[i]
	m = list_m[i]
	list_m_leaf = list_m_leaf_tot[i]
	list_p = list_p_tot[i]
	Plot.PlotPolygon(m,part)
	
	#namefile = name+'stesso_criterio_ritento_'+str(i+1)
	#Classification_BU(m,part,weight,score,namefile)

	#number_of_clusters = 3
	namefile = False#name+str(i+1)
	for number_of_clusters in range(len(list_p)):
		Plot.Plot2D(part,list_m_leaf,list_p,number_of_clusters,namefile)
	#Plot.Plot3D(list_m_leaf,list_p,number_of_clusters)#part,
	#Plot.Plot3D(list_m_leaf,list_p,part,number_of_clusters)