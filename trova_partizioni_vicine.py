

#part_polygon  ha tutto
#part[['time', 'father', 'part_number', 'leaf','brother']]  

brother = []
brother.append('nan')
for i in range(1,len(part)):
	for j in range(len(part)):
		if (part['father'].iloc[i]==part['father'].iloc[j]) and (part['part_number'].iloc[i]!=part['part_number'].iloc[j]):
			brother.append(part['part_number'].iloc[j])

part['brother'] = brother


#%%


neighbors = []
# fratelli
for i in range(len(part)):
	neighbors.append([part['brother'].iloc[i]])
	
part['neighbors'] = neighbors

#%%
	
for i in part['part_number']:
	if i%2 == 0:
		continue
	
	if (part.query('part_number=='+str(i))['leaf'].iloc[0] !=True) or (part.query('brother=='+str(i))['leaf'].iloc[0] !=True):
		
		
		part_ridotto = part.query('(part_number=='+str(i)+') or (part_number=='+str(part['brother'].iloc[i])+') or (father=='+str(i)+') or (father=='+str(part['brother'].iloc[i])+')').copy()
		part_ridotto.index = np.arange(len(part_ridotto))
 		
		for h in list(part_ridotto['part_number'].iloc[0:2]):
			part_ridotto2 = part_ridotto.query('(father=='+str(h)+') or (part_number=='+str(h)+')').copy()
			part_ridotto2.index = np.arange(len(part_ridotto2))
			controllo = []
			if len(part_ridotto2) == 1:
				continue
			for j in part_ridotto2['vertici'].iloc[1]:
				if j == part_ridotto2['vertici'].iloc[0][0]:
					#print(j)
					#print(part_ridotto2['vertici'].iloc[0][0])
					index = part.query('part_number=='+str(part_ridotto2['part_number'].iloc[1])).index
					part['neighbors'].iloc[index[0]].append(part_ridotto2['brother'].iloc[0])
					controllo.append(0)
					break
			if len(controllo)==0:
				for j in part_ridotto2['vertici'].iloc[2]:
					if j == part_ridotto2['vertici'].iloc[0][0]:
						index = part.query('part_number=='+str(part_ridotto2['part_number'].iloc[2])).index
						part['neighbors'].iloc[index[0]].append(part_ridotto2['brother'].iloc[0])	
						controllo.append(0)
						break
			if len(controllo)==0:
				index1 = part.query('part_number=='+str(part_ridotto2['part_number'].iloc[1])).index
				part['neighbors'].iloc[index1[0]].append(part_ridotto2['brother'].iloc[0])
				index2 = part.query('part_number=='+str(part_ridotto2['part_number'].iloc[2])).index
				part['neighbors'].iloc[index2[0]].append(part_ridotto2['brother'].iloc[0])

		if len(part_ridotto)==6:
			padre1 = part_ridotto['part_number'].iloc[0]
			padre2 = part_ridotto['part_number'].iloc[1]
			
			for i,j in zip([2,3],[4,5]): #figli di 0 e figli di 1
				if (padre2 in part_ridotto['neighbors'].iloc[i]) and (padre1 in part_ridotto['neighbors'].iloc[j]):
					index1 = part.query('part_number=='+str(part_ridotto['part_number'].iloc[i])).index
					part['neighbors'].iloc[index1[0]].append(part_ridotto['part_number'].iloc[j])
					index2 = part.query('part_number=='+str(part_ridotto['part_number'].iloc[j])).index
					part['neighbors'].iloc[index2[0]].append(part_ridotto['part_number'].iloc[i])
	
		
			




#%%

se il fratello del padre ha figli vicini al padre, allora dobbiamo metterli 
come vicini dei figli del padre che sono vicini al fratello del padre



#%%
part['neighbors'] = 0



neighbors = []
part_number = []

part_number.append(0)
part_number.append(1)
part_number.append(2)
neighbors.append('nan')
neighbors.append([2])
neighbors.append([1])

for i in part.query("father!='nan' and father!=0")['father'].unique():
	neigh1 = []
	neigh2 = []
	
	part_ridotto = part.query('father=='+str(i)+"or part_number=="+str(i))
	for j in part_ridotto['vertici'].iloc[1]:
		if j == part_ridotto['vertici'].iloc[0][0]:
			neigh1.append(part_ridotto['brother'].iloc[0])
			break
	for j in part_ridotto['vertici'].iloc[2]:
		if j == part_ridotto['vertici'].iloc[0][0]:
			neigh2.append(part_ridotto['brother'].iloc[0])	
			break
	if (len(neigh1)==0) and (len(neigh2)==0):
		neigh1.append(part_ridotto['brother'].iloc[0])
		neigh2.append(part_ridotto['brother'].iloc[0])
		
	neigh1.append(part_ridotto['brother'].iloc[1])
	neigh2.append(part_ridotto['brother'].iloc[2])
	
	part_number.append(part_ridotto['part_number'].iloc[1])
	part_number.append(part_ridotto['part_number'].iloc[2])
	neighbors.append(neigh1)
	neighbors.append(neigh2)
	
			
	
df = {'part_number':part_number,'neighbors':neighbors}
df = pd.DataFrame(df)	
#%%

for i in range(1,len(df)):
	for j in range(1,len(df)):
		if df['part_number'].iloc[i] in df['neighbors'].iloc[j]:
			df['neighbors'].iloc[i].append(df['part_number'].iloc[j])
df.loc[df.index,'neighbors'] = [*[list(np.unique(df['neighbors'].iloc[i])) for i in range(len(df))]]


df['leaf'] = part['leaf']




#%%			
for i in range(1,len(df)):
	for h in df['neighbors'].iloc[i]:
		for j in range(1,len(df)):
			if h in df['neighbors'].iloc[j]:
				if df['part_number'].iloc[j] not in df['neighbors'].iloc[i]:
					df['neighbors'].iloc[i].append(df['part_number'].iloc[j])
df.loc[df.index,'neighbors'] = [*[list(np.unique(df['neighbors'].iloc[i])) for i in range(len(df))]]
		
#part = pd.merge(part,df,how='left',left_on='part_number',right_on='part_number')
	






























#%%



#       trovare partizioni vicine tagli perpendicolari






def trova_part_vicine(part):


	neighbors = []
	
	for i in range(len(part.query('leaf==True'))):
		
		p = part.query('leaf==True').copy()
		p.index = np.arange(len(part.query('leaf==True')))
		
		for j in range(2):
			p['min'+str(j)+'_'+str(i)] = p['min'+str(j)].iloc[i]
			p['max'+str(j)+'_'+str(i)] = p['max'+str(j)].iloc[i]
			
		for j in range(2):	
			p=(p.eval('vicinoA'+str(j)+'_'+str(i)+' = ((min'+str(j)+'>=min'+str(j)+'_'+str(i)+') and (min'+str(j)+'<=max'+str(j)+'_'+str(i)+')) or ( (max'+str(j)+'>=min'+str(j)+'_'+str(i)+') and (max'+str(j)+'<=max'+str(j)+'_'+str(i)+'))')
		 .eval('vicinoB'+str(j)+'_'+str(i)+' = ((min'+str(j)+'_'+str(i)+'>=min'+str(j)+') and (min'+str(j)+'_'+str(i)+'<=max'+str(j)+')) or ( (max'+str(j)+'_'+str(i)+'>=min'+str(j)+') and (max'+str(j)+'_'+str(i)+'<=max'+str(j)+'))'))
			
			#p=p.query('vicino'+str(j)+'_'+str(i)+'==True')
			p=p.query('(vicinoA'+str(j)+'_'+str(i)+'==True) or (vicinoB'+str(j)+'_'+str(i)+'==True)')
			
			
		p = p.drop(i)
		
		neighbors.append(list(p['part_number']))
	
	
	
	df={'part_number':part.query('leaf==True')['part_number'],'neighbors':neighbors}
	df=pd.DataFrame(df)
	df.index = np.arange(len(df))
	



	return df
# per più di due dimensioni?
# come generalizzarla a partizione con tagli non regolari?





#  calcolo varianza per partizioni vicine 


def calcolo_varianza_part_vicine(data,i,j):
	
	
	
	data1 = data.query('part_number=='+str(i))
	data2 = data.query('part_number=='+str(j))
	
	pd1 = pdist(data1)
	pd2 = pdist(data2)
	
	pd = pdist(data)
	pd12 = np.hstack([pd1, pd2])
	
	var_part_unica = np.var(pd)
	var1 = np.var(pd1)
	var2 = np.var(pd2)
	
	#var_part_separate = np.var(pd12)
	#score_1 = np.abs(np.log(np.var(pd12)/np.var(pd)))
	#score_2 = np.abs(np.log(np.var(pd1)/np.var(pd2)))
	#print(score_1>score_2)

	
	return var_part_unica,var1,var2 #score_1,score_2, var_part_unica,var_part_separate





part_vicine = trova_part_vicine(part)
punti = AssignPartition(X,part)

#separazione_corretta = [] 


part1 = []
part2 = []
score_1 = []
score_2 = []
#v_unica = []
#v_sep = []


for i in part_vicine['part_number']:
	
	for j in list(part_vicine[part_vicine['part_number']==i]['neighbors'])[0]:
		part2.append(j)
		part1.append(i)
		
		p = punti.query('(part_number=='+str(i)+') or (part_number=='+str(j)+')').copy()
		
		
		var_part_unica,var1,var2 = calcolo_varianza_part_vicine(punti,i,j)
		#s1,s2,v1,v2 = calcolo_varianza_part_vicine(punti,i,j)
		#score_1.append(s1)
		#score_2.append(s2)
		#v_unica.append(v1)
		#v_sep.append(v2)
		
		
df = {'part1':part1,'part2':part2,'var_part_unica':var_part_unica,'var1':var1,'var2':var2}#'score_1':score_1,'score_2':score_2}		
df = pd.DataFrame(df)		#'v_unica':v_unica,'v_sep':v_sep}#
		#if s1 > s2:
		#	separazione_corretta.append(True)
		#else:
		#	separazione_corretta.append(False)
		

		
		
	
#conto_punti = punti.query('part_number=='+str(i)).count()[0]


