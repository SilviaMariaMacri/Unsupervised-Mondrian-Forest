# cd "C:\Users\silvi\Desktop\Fisica\TESI\Tesi"
import time

start = time.time()
end = time.time()
print(end - start)


#%% salvo dati

import json


def SaveMondrianOutput(namefile,part,m):
	#part
	part.to_json(namefile+'_part.json')
	#m
	lista = list(np.array(m,dtype=object)[:,2])
	for i in lista:
		i.columns = i.columns.astype(str)
	with open(namefile+'_m.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in lista]))
	return
	


# Polytope dimensione generica + classificazione 

t0 = 0
lifetime = 10
dist_matrix = DistanceMatrix(X)
number_of_iterations = 10
#name = 'makemoons3D2_lambda10_'
#prova a diminuire esponenteeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
#eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
for i in range(number_of_iterations):

	#i=1
	m_i,part_i = Mondrian(X,t0,lifetime,dist_matrix)
	namefile = name+str(i+1)
	SaveMondrianOutput(namefile,part_i,m_i)

	part = json.load(open(namefile+'_part.json','r'))
	part = pd.DataFrame(part)
	m = json.load(open(namefile+'_m.json','r'))
	#PlotPolygon(m,part)

	#tagli_paralleli = False #True#,False
	score = 'min' #'var','centroid'
	weight = 'diff_min' #'var_ratio','ratio_centroid','diff_centroid',
	#Classification(part,m,X,namefile,score,weight,tagli_paralleli)
	#Classification_BU(m,part,weight,score,namefile)
	list_m_leaf,list_p = Classification_BU(m,part,weight)
	list_p.reverse()
	list_m_leaf.reverse()
	
	with open(namefile+'_p.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in list_p]))
	with open(namefile+'_m_leaf.json', 'w') as f:
		f.write(json.dumps([df for df in list_m_leaf]))
	
#togli X da classification


#%% leggo file .json


number_of_iterations = 10

#name = 'cerchi3D1_lambda6'
#name = 'cerchi3D1_'
#name = 'cerchi3D1_lambda25_'

#name = 'circles3D_'
#name = 'circles3D_exp50_'
#name = 'circles3D_exp70_'
#name = 'circles3D_lambda25_'

#name= 'semisfera_lambda25_'

#name= 'cilindro_lambda25_'

#name='circles3D_' # da 2 a 10
#namefile='makemoons+blob3D_1'

#name='makemoons3D2_lambda50_'




list_part = []
list_m = []

list_p_tot = []
list_m_leaf_tot = []

#list_class = []
#list_conn_comp = []


for i in range(number_of_iterations):

	namefile = name+str(i+1)
	
	part = json.load(open(namefile+'_part.json','r'))
	part = pd.DataFrame(part)
	m = json.load(open(namefile+'_m.json','r'))
	list_part.append(part)
	list_m.append(m)
	
	list_p = json.load(open(namefile+'_p.json','r'))
	list_m_leaf = json.load(open(namefile+'_m_leaf.json','r'))
	list_p_tot.append(list_p)
	list_m_leaf_tot.append(list_m_leaf)	
	
	
	'''
	#namefile = name+'stesso_criterio_ritento_'+str(i+1)
	list_p = []
	list_m_leaf = []
	for l in range(1000):
		try:
			p = json.load(open(namefile+'_p_'+str(l)+'.json','r'))
			p = pd.DataFrame(p)
			m_leaf = json.load(open(namefile+'_m_leaf_'+str(l)+'.json','r'))
			
			list_p.append(p)
			list_m_leaf.append(m_leaf)
		except FileNotFoundError:
			break
	list_p.reverse()
	list_m_leaf.reverse()
	list_p_tot.append(list_p)
	list_m_leaf_tot.append(list_m_leaf)
	
	#classified_data = json.load(open(namefile+'_list_class.json','r'))
	#conn_comp = json.load(open(namefile+'_conn_comp.json','r'))
	#list_class.append(classified_data)
	#list_conn_comp.append(conn_comp)
	'''



#%%  grafico compatibilità classificazioni

#name_file = 'makemoons_2_'
#coeff_medio = ClassificationScore(list_class,name_file)


class_data_tot = []
for i in range(len(list_m_leaf_tot)):
	list_m_leaf = list_m_leaf_tot[i].copy()
	classified_data = AssignClass_BU(list_m_leaf)
	class_data_tot.append(classified_data)

ClassificationScore_BU(class_data_tot,name)


#%% confronto classificazione vera

coeff_tot = []
coeff_medio_tot = []
max_number = 15
number_of_clusters_true = 2
for number_of_clusters in range(1,max_number):
	print(number_of_clusters)
	coeff = ConfrontoClasseVera(class_data_tot,y,number_of_clusters)
	#if number_of_clusters == number_of_clusters_true:
	print(coeff)
	#print('min: ',np.min(coeff))
	#print('max: ',np.max(coeff))
	coeff_medio = np.mean(coeff)
	coeff_medio_tot.append(coeff_medio)
	coeff_tot.append(coeff)
	
fig,ax = plt.subplots()
ax.plot(np.arange(1,max_number),coeff_medio_tot)
ax.scatter(np.arange(1,max_number),coeff_medio_tot)
for i in range(len(np.array(coeff_tot).T)):
	#fig,ax = plt.subplots()
	ax.plot(np.arange(1,max_number),np.array(coeff_tot).T[i],alpha=0.2)

#%% grafici

#name = 'makeblobs_3D_'
number_of_iterations = 10
for i in range(number_of_iterations):
	print(i)
	
	#i=2
	part = list_part[i]
	m = list_m[i]
	list_m_leaf = list_m_leaf_tot[i]
	list_p = list_p_tot[i]
	#PlotPolygon(m,part)
	
	#namefile = name+'stesso_criterio_ritento_'+str(i+1)
	#Classification_BU(m,part,weight,score,namefile)

	number_of_clusters = 2
	#namefile = False#name+str(i+1)
	#for number_of_clusters in range(len(list_p)):
	#Plot2D(part,list_m_leaf,list_p,number_of_clusters,namefile)
	Plot3D(part,list_m_leaf,list_p,number_of_clusters)
	
	#classified_data = list_class[i]
	#conn_comp = list_conn_comp[i]
	#for number_of_clusters in range(len(conn_comp)):
	#name_file = False#'plot_3clusters_MP_'+str(i+1)	
	#PlotClass_2D(m,part,conn_comp,number_of_clusters,name_file)
	#PlotClass_3D(m,part,conn_comp,number_of_clusters)



#%% confronto altri metodi di clustering

from sklearn.cluster import KMeans,AgglomerativeClustering,Birch,SpectralClustering,mean_shift
from sklearn.metrics.cluster import fowlkes_mallows_score

n_clusters = 2
title = ['KMeans','AgglomerativeClustering','Birch','SpectralClustering']#,'MeanShift']
labels = []

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
labels.append(kmeans.labels_)
print('kmeans: ',fowlkes_mallows_score(y,kmeans.labels_))

AC = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
labels.append(AC.labels_)
print('AgglomerativeClustering: ',fowlkes_mallows_score(y,AC.labels_))

B = Birch(n_clusters=n_clusters).fit(X)
labels.append(B.labels_)
print('Birch: ',fowlkes_mallows_score(y,B.labels_))

SC = SpectralClustering(n_clusters=n_clusters,random_state=0).fit(X)#,assign_labels='discretize'
labels.append(SC.labels_)
print('SpectralClustering: ',fowlkes_mallows_score(y,SC.labels_))

#MS = mean_shift(X)
#labels.append(MS[1])
#print('MeanShift: ',fowlkes_mallows_score(y,MS[1]))

for i in range(len(labels)):
	
	data = {'X0':X[:,0],'X1':X[:,1],'X2':X[:,2],'y':list(labels[i])}
	
	data=pd.DataFrame(data)
	
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter3D(np.array(data.query('y==0'))[:,0],np.array(data.query('y==0'))[:,1],np.array(data.query('y==0'))[:,2],alpha=0.5,color='b')
	ax.scatter3D(np.array(data.query('y==1'))[:,0],np.array(data.query('y==1'))[:,1],np.array(data.query('y==1'))[:,2],alpha=0.5,color='orange')
	ax.scatter3D(np.array(data.query('y==2'))[:,0],np.array(data.query('y==2'))[:,1],np.array(data.query('y==2'))[:,2],alpha=0.5,color='g')
	ax.set_title(title[i])



























#%% Polygon 2D

t0=0
lifetime=2
#dist_matrix = DistanceMatrix(X)
m,box,part=MondrianPolygon(X,t0,lifetime,dist_matrix)
PlotPolygon(X,part)

#[['time', 'father', 'part_number', 'neighbors', 'leaf']]

#%% Unsupervised

t0=0
lifetime=2.5
part = MondrianUnsupervised(X,t0,lifetime)
#number_iterations = 10
#X_part = AssignPartition(X,part)
#df =trova_part_vicine(part)
#matrix,points_with_index,part_tot = MondrianIterator(number_iterations,X,t0,lifetime)
#PartitionPlot(X,y,part_tot)
PlotPolygon(X,part)

#%% Supervised

t0=0
lifetime=2

part = MondrianSupervised(X_train,y_train,t0,lifetime)
#part_with_counts = Count(X_train,y_train,part)
#accuracy,cl = AssignClass(X_test,y_test,part_with_counts)
PartitionPlot(X_train,y_train,part)
