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
lifetime = 1.5
#dist_matrix = DistanceMatrix(X)
number_of_iterations = 1
name = 'moons1_METRIC_'
#prova a diminuire esponenteeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
#eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
for i in range(number_of_iterations):

	#i=1
	case = 'min corr' #'min','variance','centroid diff','centroid ratio'
	m_i,part_i = Mondrian(X,t0,lifetime,dist_matrix,case,exp)
	namefile = name+str(i+1)
	SaveMondrianOutput(namefile,part_i,m_i)

	part = json.load(open(namefile+'_part.json','r'))
	part = pd.DataFrame(part)
	m = json.load(open(namefile+'_m.json','r'))
	PlotPolygon(m,part)

	#tagli_paralleli = False #True#,False
	score = 'min' #'var','centroid'
	weight = 'diff_min' #'var_ratio','ratio_centroid','diff_centroid',
	#Classification(part,m,X,namefile,score,weight,tagli_paralleli)
	#Classification_BU(m,part,weight,score,namefile)
	list_m_leaf,list_p = Classification_BU(m,part,metric)
	list_p.reverse()
	list_m_leaf.reverse()
	
	with open(namefile+'_p.json', 'w') as f:
		f.write(json.dumps([df.to_dict() for df in list_p]))
	with open(namefile+'_m_leaf.json', 'w') as f:
		f.write(json.dumps([df for df in list_m_leaf]))
	
#togli X da classificationcd


#%% leggo file .json


number_of_iterations = 20
name = 'iris_10'


list_part = []
list_m = []

list_p_tot = []
list_m_leaf_tot = []

#list_class = []
#list_conn_comp = []


for i in range(number_of_iterations):

	namefile = name#+str(i+1)
	
	part = json.load(open(namefile+'_part.json','r'))
	part = pd.DataFrame(part)
	m = json.load(open(namefile+'_m.json','r'))
	list_part.append(part)
	list_m.append(m)
	
	'''
	list_p = json.load(open(namefile+'_p.json','r'))
	list_m_leaf = json.load(open(namefile+'_m_leaf.json','r'))
	list_p_tot.append(list_p)
	list_m_leaf_tot.append(list_m_leaf)	
	print(len(pd.DataFrame(list_p[-1])))'''
		
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
	'''
	
	'''
	classified_data = json.load(open(namefile+'_list_class.json','r'))
	conn_comp = json.load(open(namefile+'_conn_comp.json','r'))
	list_class.append(classified_data)
	list_conn_comp.append(conn_comp)
	'''



#%% aggiunta per grafico score determinazione classi diverso esponente

#moons2D
#exp = ['_exp1_','_exp2_','_exp2.5_','_exp3_','_exp4_','_exp5_','_exp20_','_exp35_','_','_exp65_']#,'_exp80_','_exp100_','_exp120_']

#cerchi3D2
#lifetime = ['_lambda50','_','_lambda50','_lambda200','_lambda200','_lambda200']
#exp = ['_exp5_','exp10_','_exp20_','_exp5_','_exp10_','_exp20_']

#blobs3D
#lifetime = '_lambda500'
#exp =['_exp20_','_exp20_']#'_exp5_',
#metric = ['min_corr','variance']#'min_corr',

#cilindro
#lifetime = ['_lambda50','_lambda200','_lambda1000','_lambda50']
#exp = ['_exp5_','_exp5_','_exp5_','_exp10_']
#metric = 'min_corr'

#cilindro_nuovo
#lifetime = ['_lambda50','_lambda200']
#exp = '_exp5_'
#metric = 'min_corr'

#2D confronto metriche 


metric = 'min_corr'

lifetime = ['_lambda50','_lambda200']
exp = '_exp5_'

FMS_tot = []
FMS_medio_tot = []
FMS_std_tot = []
c_mean = []
c_std = []
c_tot = []
for j in range(len(metric)):
	print(j+1)
   

	number_of_iterations = 20
	#name = 'varied_'+metric[j]+'_'
	name = 'sfera'+lifetime[j]+exp+metric+'_'
	
	list_part = []
	list_m = []
	list_p_tot = []
	list_m_leaf_tot = []
	
	for i in range(number_of_iterations):
	
		namefile = name+str(i+1)
		
		#part = json.load(open(namefile+'_part.json','r'))
		#part = pd.DataFrame(part)
		#m = json.load(open(namefile+'_m.json','r'))
		#list_part.append(part)
		#list_m.append(m)
		
	
		list_p = json.load(open(namefile+'_p.json','r'))
		list_m_leaf = json.load(open(namefile+'_m_leaf.json','r'))
		list_p_tot.append(list_p)
		list_m_leaf_tot.append(list_m_leaf)	
	

	class_data_tot = []
	for i in range(len(list_m_leaf_tot)):
		list_m_leaf = list_m_leaf_tot[i].copy()
		classified_data = AssignClass_BU(list_m_leaf)
		class_data_tot.append(classified_data)
	
	c_mean_i,c_std_i,c_tot_i = ClassificationScore_BU(class_data_tot,False)
	c_mean.append(c_mean_i)
	c_std.append(c_std_i)
	c_tot.append(c_tot_i)
	
	
	
	number_of_clusters = 3
	FMS = ConfrontoClasseVera(class_data_tot,y,number_of_clusters)
	FMS_medio = np.mean(FMS)
	FMS_std = np.std(FMS)
	FMS_tot.append(FMS)
	FMS_medio_tot.append(FMS_medio)
	FMS_std_tot.append(FMS_std)
	
	
	
	
#%%
color = ['d','b','orange','g','r','purple','cyan']
#color = ['b','orange','g','b','orange','g']	
#label = ['5','20','35','50','65']#,'80','100','120']'1','2','2.5','3','4',	
#label = ['lambda50_exp5','lambda50_exp10','lambda50_exp20','lambda200_exp5','lambda200_exp10','lambda200_exp20']
#label = ['min_exp5','min_exp20','variance_exp20']
#label = ['lambda50_exp5','lambda200_exp5','lambda1000_exp5','lambda50_exp10']
label = ['a','min_corr','variance']
'''
fig,ax = plt.subplots()
for i in range(len(label)):
	ax.plot(np.arange(2,len(c_mean[i])+1),c_mean[i][1:],label=label[i],alpha=0.5,color=color[i])
	ax.scatter(np.arange(2,len(c_mean[i])+1),c_mean[i][1:],s=10,alpha=0.7)
	ax.legend()#title='Exp'
	#plt.savefig('moons2D_coeff_exp'+label[i])
	ax.set_xlabel('Number of Clusters')
	ax.set_ylabel('Adjusted Mutual Information')
'''	

fig,ax = plt.subplots()
for i in range(1,len(label)):
	#if (i==2) or (i==5):
	ax.plot(np.arange(2,len(c_mean[i])+1),c_mean[i][1:],label=label[i],color=color[i],linewidth=0.7)
	ax.scatter(np.arange(2,len(c_mean[i])+1),c_mean[i][1:],s=10,color=color[i])
	ax.fill_between(np.arange(2,len(c_mean[i])+1), c_mean[i][1:]-c_std[i][1:]/2, c_mean[i][1:]+c_std[i][1:]/2,alpha=0.2,color=color[i])
	ax.legend()#title='Exp'		
	#plt.savefig('moons2D_coeff_exp'+label[i])
	ax.set_xlabel('Number of Clusters')
	ax.set_ylabel('Adjusted Mutual Information')
	

#%%
df = {'lambda50':c_std[0],'lambda200':c_std[1]}#,'min':c_mean[2],'min_corr':c_mean[3]}#,'lambda200_exp10':c_mean[4],'lambda200_exp20':c_mean[5]}	
df = pd.DataFrame(df)
df.to_csv('sfera_AMIstd.txt',sep='\t',index=False)

#%%

df = pd.DataFrame(c_tot[1])
df.to_csv('sfera_AMI_completo_lambda200.txt',sep='\t',index=False)

#AMI_completo => colonne = numero di cluster, righe =coppie di risultati
#df2.eval('due_minore_di_tre = due<tre',inplace=True)

#%%  grafico compatibilità classificazioni

#name_file = 'makemoons_2_'
#coeff_medio = ClassificationScore(list_class,name_file)


class_data_tot = []
for i in range(len(list_m_leaf_tot)):
	list_m_leaf = list_m_leaf_tot[i].copy()
	classified_data = AssignClass_BU(list_m_leaf)
	class_data_tot.append(classified_data)

c_mean_singolo,c_std_singolo = ClassificationScore_BU(class_data_tot,False)





#%% confronto classificazione vera

coeff_tot = []
coeff_medio_tot = []
#max_number = 15
#number_of_clusters_true = 2
number_of_clusters = 3
#for number_of_clusters in range(1,max_number):
#	print(number_of_clusters) 
coeff = ConfrontoClasseVera(class_data_tot,y,number_of_clusters)
	#if number_of_clusters == number_of_clusters_true:
print(coeff)
	#print('min: ',np.min(coeff))
	#print('max: ',np.max(coeff))
coeff_medio = np.mean(coeff)
print('coeff medio: ', coeff_medio)
coeff_std = np.std(coeff)
print('coeff std: ',coeff_std)
	#coeff_medio_tot.append(coeff_medio)
	#coeff_tot.append(coeff)
#%%	
fig,ax = plt.subplots()
ax.plot(np.arange(1,max_number),coeff_medio_tot)
ax.scatter(np.arange(1,max_number),coeff_medio_tot)
for i in range(len(np.array(coeff_tot).T)):
	#fig,ax = plt.subplots()
	ax.plot(np.arange(1,max_number),np.array(coeff_tot).T[i],alpha=0.2)

#%% grafici

#name = 'makeblobs_3D_'
#number_of_iterations = 10
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

	number_of_clusters = 3
	namefile = False#name+str(i+1)
	#for number_of_clusters in range(len(list_p)):
	#Plot2D(part,list_m_leaf,list_p,number_of_clusters,namefile)
	Plot3D(list_m_leaf,list_p,number_of_clusters)#part,

	#classified_data = list_class[i]
	#conn_comp = list_conn_comp[i]
	#for number_of_clusters in range(len(conn_comp)):
	#name_file = False#'plot_3clusters_MP_'+str(i+1)	
	#PlotClass_2D(m,part,conn_comp,number_of_clusters,name_file)
	#PlotClass_3D(m,part,conn_comp,number_of_clusters)



#%% confronto altri metodi di clustering

from sklearn.cluster import KMeans,DBSCAN,SpectralClustering#,AgglomerativeClustering,Birch,mean_shift
from sklearn.metrics.cluster import fowlkes_mallows_score

n_clusters = 3
title = ['KMeans','DBSCAN','SpectralClustering']#,'MeanShift']'AgglomerativeClustering','Birch',
labels = []
'''
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
#
#%%
print('kmeans: ')
print(FMS_kmeans)
print('DBSCAN: ')
print(FMS_dbscan)
print('Spectral clustering: ')
print(FMS_spectral)


#%%
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






#%%
for i in range(len(labels)):
#	i=1
	
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
























#%% Polygon 2D

t0=0
lifetime=2
#dist_matrix = DistanceMatrix(X)
m,box,part=MondrianPolygon(X,t0,lifetime,dist_matrix)
PlotPolygon(X,part)

#[['time', 'father', 'part_number', 'neighbors', 'leaf']]

#%% Unsupervised

t0=0
lifetime=6
for i in range(20):
	m,part = MondrianUnsupervised(X,t0,lifetime)
	part.to_json('circles_tagli_perpendicolari_'+str(i)+'_part.json')
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
