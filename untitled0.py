a,b = -0.5,-1.25 
theta = np.radians(-45)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s, (a-a*c+b*s)), (s, c, (b-b*c-a*s))))


f=pd.DataFrame(X)
f[2] = 1
X = np.array(f)

x=X.copy()
X=[]
for i in range(len(x)):
	X.append(list(R@x[i]))
X = np.array(X)

fig,ax = plt.subplots()
ax.scatter(X[:,0],X[:,1])

X_ruotato = X.copy()


#%%

#gaussiane in diagonale

		
a = 50

# diagonale
b1 = 0.07
b2 = 0.06
#verticali
#b1 = 0.01
#b2 = 0

mean1 = (0, 0)
cov1 = [[b1,b2+0.1], [b2,b1]]
np.random.seed(30)
x1 = np.random.multivariate_normal(mean1, cov1, a)

#mean2 = (1,0)
#mean2 = (0.6, -0.7) #paralleli
mean2 = (0.8,-0.6)
cov2 = [[b1,b2], [b2,b1]]
np.random.seed(30)
x2 = np.random.multivariate_normal(mean2, cov2, a)


mean3 = (-0.8,-1.6)
cov3 = [[0.02,0], [0,0.02]]
np.random.seed(7)
x3 = np.random.multivariate_normal(mean3, cov3, a)



mean4 = (0,-2.5)
cov4 = [[0.04,0], [0,0.1]]
np.random.seed(8)
x4 = np.random.multivariate_normal(mean4, cov4, a)


mean5 = (2.5,-2.5)
cov5 = [[0.01,b1+0.1], [b1+0.1,0.01]]
np.random.seed(50)
x5 = np.random.multivariate_normal(mean5, cov5, a)




			
X = np.vstack([x1,x2,x3,x4,x5])

			
fig,ax = plt.subplots()
#ax.scatter(x1[:,0],x1[:,1],color='b')
#ax.scatter(x2[:,0],x2[:,1],color='b')
ax.scatter(X[:,0],X[:,1])




X_originale = X.copy()





#%%


mean1 = (0, 0)
cov1 = [[b,0], [0,b]]
np.random.seed(0)
x1 = np.random.multivariate_normal(mean1, cov1, a)

mean2 = (1,0)
cov2 = [[b,0], [0,b]]
np.random.seed(1)
x2 = np.random.multivariate_normal(mean2, cov2, a)


X = np.vstack([x1,x2])
fig,ax = plt.subplots()
ax.scatter(X[:,0],X[:,1])




#%% valutare tagli più probabili  varianzaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
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
			