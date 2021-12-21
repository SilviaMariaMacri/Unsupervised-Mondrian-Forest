metrics

json.load
json.dump




valutare tagli più probabili  varianzaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
# plotta dati separati
'''
dist_matrix_ridotta = dist_matrix[['index1', 'index2', 'dist', 'magnitude_norm_vect', 'index_norm_vect',
       'x1_0', 'x1_1', 'x2_0', 'x2_1', 'x_cut_0', 'x_cut_1', 'norm_vect_0',
       'norm_vect_1']].copy()

dist_matrix_ridotta = dist_matrix_ridotta.drop_duplicates()


matrice = pd.merge(data_pair,dist_matrix_ridotta,left_on=['index1','index2'],right_on=['index1','index2'])

'''

data_pair['var_ratio*50'] =data_pair['var_ratio']**50
data_pair['prob'] = data_pair['var_ratio*50']/data_pair['var_ratio*50'].sum()



matrice = pd.merge(data_pair,dist_matrix,left_on=['index1','index2'],right_on=['index1','index2'])

#matrice = matrice.sort_values(by='var_ratio',ascending=False)

data_pair = data_pair.sort_values(by='var_ratio',ascending=False)
data_pair.index = np.arange(len(data_pair))


#%%

for i in range(400,600):
	scelgo_cut = matrice.query('index1=='+str(data_pair['index1'].iloc[i])+' and index2=='+str(data_pair['index2'].iloc[i])).copy()
	data1 = scelgo_cut.query('dist_point_cut>0').copy()
	data2 = scelgo_cut.query('dist_point_cut<0').copy()
	
	fig,ax = plt.subplots()
	ax.scatter(data1['point_0'],data1['point_1'],color='b')
	ax.scatter(data2['point_0'],data2['point_1'],color='orange')




#%%



#righe con var_Ratio uguale a valore massimo
#data_pair_primi_20 = data_pair.sort_values(by='var_ratio',ascending=False).query('var_ratio=='+str(data_pair['var_ratio'].max()))
#primi 20
data_pair_primi_20 = data_pair.sort_values(by='var_ratio',ascending=False).iloc[0:20]
data_pair_primi_20.index=np.arange(len(data_pair_primi_20))

df=[]
for i in range(len(data_pair_primi_20)):
    df.append(dist_matrix.query('index1=='+str(data_pair_primi_20['index1'].iloc[i])+' and index2 =='+str(data_pair_primi_20['index2'].iloc[i])).iloc[0])

fig,ax = plt.subplots()
for i in range(len(data_pair_primi_20)):
    ax.plot(df[i][['x1_0','x2_0']],df[i][['x1_1','x2_1']])
ax.scatter(x1[:,0],x1[:,1],color='b')
ax.scatter(x2[:,0],x2[:,1],color='b')





#%%


for i in part:
	i.to_csv('100iterazioni_NOheader.txt',mode='a',sep='\t',header=False)



#%%

#part = pd.read_csv('98iterazioniNOheader.txt',sep='\t')



color=cm.rainbow(np.linspace(0,1,len(part)))
		
fig, ax = plt.subplots()
		
for j,c in zip(part[61:62],color):

	part_provvisoria = j.iloc[0:2].copy()#
	leaf = []
	for i in range(20):#len(j)
		if part_provvisoria['part_number'].iloc[i] not in part_provvisoria['father'].unique():
			leaf.append(True)
		else:
			leaf.append(False)
		
			
	part_provvisoria['leaf'] = leaf
	
	
			
	for i in range(len(part_provvisoria.query('leaf==True'))):
		box_new = part_provvisoria.query('leaf==True')['box'].iloc[i]
		p = Polygon(box_new, facecolor = 'none', edgecolor='b',alpha=0.5)
		ax.add_patch(p)
				
			
ax.scatter(X[:,0],X[:,1],s=10,alpha=0.5)
			
plt.show()
			










#%%     dendrogramma


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





#%%  area poligono


box = box[0]


def PolyArea(box):
	
	df_box = pd.DataFrame(box)
	x = df_box[0]
	y = df_box[1]
    
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

print(PolyArea(box))






