n = np.geomspace(0.6, 2, 20)
e = np.geomspace(0.002, 0.5, 10)



media = []
err = []
r = []
d = []

count=0
for m in n:



	for b in e:
		count=count+1
		
		a = 50
		
		mean1 = (0,0)
		cov1 = [[b,0], [0,b]]
		np.random.seed(0)
		x1 = np.random.multivariate_normal(mean1, cov1, a)
		
		mean2 = (m,m)
		cov2 = [[b,0], [0,b]]
		np.random.seed(1)
		x2 = np.random.multivariate_normal(mean2, cov2, a)
		
		X = np.vstack([x1,x2])
		
		ratio,difference = Centroid(x1,x2,X)
		r.append(ratio)
		d.append(difference)
		media.append(m)
		err.append(b)
		
		
		fig,ax = plt.subplots()
		ax.scatter(X[:,0],X[:,1])
		ax.set_title('m='+str(m)+' b='+str(b))
		plt.savefig(str(count))
		
		
		
		
df = {'ratio':r,'diff':d,'m':media,'b':err}
df = pd.DataFrame(df)
df.index = np.arange(len(df)+1)[1:]

indice_ratio = df.sort_values(by='ratio',ascending=False).index
indice_diff = df.sort_values(by='diff',ascending=False).index
df_sorted_ratio = df.sort_values(by='ratio',ascending=False)
#df_sorted_ratio.index = np.arange(len(df_sorted_ratio))
df_sorted_diff = df.sort_values(by='diff',ascending=False)
#df_sorted_diff.index = np.arange(len(df_sorted_diff))

df['diff_elevata']=df['diff']*10

fig,ax=plt.subplots()
#ax.plot(df.index,df['ratio'],label='ratio')
#ax.plot(df.index,df['diff'],label='diff')
sns.lineplot(data=df,#.query('m==0.6'), 
			 x='b', y='ratio', hue='m')
sns.lineplot(data=df,#.query('m==0.6'), 
			 x='b', y='diff_elevata', hue='m')

