# cd "C:\Users\silvi\Desktop\Fisica\TESI\Tesi"

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

import seaborn as sns


#%% data



dat = datasets.make_moons(n_samples=150,noise=0.07)
iris = datasets.load_iris()
'''
#iris
data = pd.DataFrame(iris.data)
data[[0,1,2]]
'''



#%% 2D
dat = datasets.make_circles(n_samples=300,noise=0.05,random_state=0,factor=0.5)
#make_moons
X = dat[0]
y = dat[1]


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)




#%% 3D


#make_moons
data = pd.DataFrame(dat[0])

altra_dim = np.random.normal(0, 1, len(data))
data[2] = altra_dim
data['class']=dat[1]


X = np.array(data[[0,1,2]])
y = dat[1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)



#%%  Unsupervised

t0=0
lifetime=2

number_iterations = 10



part = MondrianUnsupervised(X,t0,lifetime)
PartitionPlot(X,y,part)

#matrix,points_with_index,part_tot = MondrianIterator(number_iterations,X,t0,lifetime)
#PartitionPlot(X,y,part_tot)


#%% Supervised

t0=0
lifetime=2

part = MondrianSupervised(X_train,y_train,t0,lifetime)
#part_with_counts = Count(X_train,y_train,part)
#accuracy,cl = AssignClass(X_test,y_test,part_with_counts)
PartitionPlot(X_train,y_train,part)



#%%  clustermap



row_colors = []
for i in range(len(y)):
	if y[i]==0:
		row_colors.append('b')
	else:
		row_colors.append('orange')

sns.clustermap(matrix, row_colors=row_colors)




#%%

#ok per 1
sns.clustermap(matrix, row_colors=row_colors, method='ward')
#sns.clustermap(matrix, row_colors=row_colors, method='weighted', metric='correlation')
#sns.clustermap(matrix, row_colors=row_colors, method='weighted', metric='cosine')


#sns.clustermap(matrix, row_colors=row_colors, method='single', metric='cityblock')







#%%


import scipy.cluster.hierarchy as sch


# retrieve clusters using fcluster 
d=sch.distance.pdist(matrix,metric='euclidean')#euclidean,correlation
L=sch.linkage(d, method='ward')#ward,weighted
# 0.2 can be modified to retrieve more stringent or relaxed clusters
#clusters=sch.fcluster(L, 0.9*d.max(), 'distance')
clusters=sch.fcluster(L,2, 'maxclust')

# clusters indicices correspond to incides of original df
classificazione_cluster = []
for i,cluster in enumerate(clusters):
	#print(matrix.index[i], cluster)
	if cluster==1:
		classificazione_cluster.append(cluster)
	else:
		classificazione_cluster.append(0)




#PartitionPlot(X,classificazione_cluster,part)

#   grafico confronto classificazioni


data2 = pd.DataFrame(X)
data2['class'] = y
data2['class_cluster'] = classificazione_cluster
	
	
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
	
		
#ax.scatter(data2[data2['class']==0][0],data2[data2['class']==0][1], facecolors='none', edgecolors='b')
#ax.scatter(data2[data2['class']==1][0],data2[data2['class']==1][1], facecolors='none', edgecolors='orange')

ax.scatter(data2[data2['class_cluster']==0][0],data2[data2['class_cluster']==0][1], color='b',alpha=0.3)
ax.scatter(data2[data2['class_cluster']==1][0],data2[data2['class_cluster']==1][1], color='orange',alpha=0.3)
	
	
	
plt.show()





#%%





#d=sch.distance.pdist(matrix,metric='euclidean')
#L=sch.linkage(d, method='ward')


# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
sch.dendrogram(
    L,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()









#%%

#method
single
complete
average
weighted
centroid   Eucl
median  Eucl
ward  Eucl

#metric
matching non usare
sqeuclidean




#%%


fig,ax = plt.subplots()

ax.plot(list(part.query('dim==0')['part_number']),list(part.query('dim==0')['distance']))
ax.plot(list(part.query('dim==1')['part_number']),list(part.query('dim==1')['distance']))



#%%

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

for i in range(len(part[part['dim']==0])):
	ax.vlines(part.query('dim==0')['x'].iloc[i],part.query('dim==0')['min1'].iloc[i],part.query('dim==0')['max1'].iloc[i],color='b')#, linestyle = '--', color=color[i], linewidth=1) # - QD_peaks[i][1]
	ax.text(part.query('dim==0')['x'].iloc[i],part.query('dim==0')['min1'].iloc[i],part.query('dim==0')['father'].iloc[i],color='b')#,  fontsize=12, color=color[i])



for i in range(len(part[part['dim']==1])):
	ax.hlines(part.query('dim==1')['x'].iloc[i],part.query('dim==1')['min0'].iloc[i],part.query('dim==1')['max0'].iloc[i],color='orange')#, linestyle = '--', color=color[i], linewidth=1) # - QD_peaks[i][1]
	ax.text(part.query('dim==1')['min0'].iloc[i],part.query('dim==1')['x'].iloc[i],part.query('dim==1')['father'].iloc[i])#,  fontsize=12, color=color[i])


