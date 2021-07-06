

N = [50,100,150,200] # numerosità
n = np.geomspace(0.01, 1, 11) # rumore
#array([0.01      , 0.01584893, 0.02511886, 0.03981072, 0.06309573,
#       0.1       , 0.15848932, 0.25118864, 0.39810717, 0.63095734,
#       1.        ])
#%%
for a in N:
		
	
	for b in n:
		
		
		#%%   gaussiane in diagonale
		
		
			a = 100
			#b = n[0]
		
			mean1 = (0, 0)
			cov1 = [[0.07,0.06], [0.06,0.07]]
			x1 = np.random.multivariate_normal(mean1, cov1, a)
			#x1=pd.DataFrame(x1)
			#x1['cl']=0
				
			mean2 = (0.3, -0.7)
			cov2 = [[0.07,0.06], [0.06,0.07]]
			x2 = np.random.multivariate_normal(mean2, cov2, a)
			#x2=pd.DataFrame(x2)
			#x2['cl']=1
			
			
			fig,ax = plt.subplots()
			ax.scatter(x1[:,0],x1[:,1],color='b')
			ax.scatter(x2[:,0],x2[:,1],color='b')
			
			X = np.vstack([x1,x2])
			


#%%


for i in part:
	i.to_csv('100iterazioni_NOheader.txt',mode='a',sep='\t',header=False)











#%%

color=cm.rainbow(np.linspace(0,1,len(part)))
		
fig, ax = plt.subplots()
		
for j,c in zip(part[61:62],color):

	part_provvisoria = j.iloc[0:20].copy()#
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
			