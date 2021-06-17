    \item hanno lo stesso padre
    \item se una delle due partizioni che hanno lo stesso padre ha 
	un lato uguale al primo lato del padre e l'altra partizione non ce l'ha, 
	allora la prima è vicina al fratello del padre, mentre la seconda no
    \item se nessuna delle due partizioni che hanno lo stesso padre ha un 
	lato uguale al primo lato del padre, allora sono entrambe vicine al 
	fratello del padre

#%%

for j in range(i-1,i+2):
		if part['father'].iloc[i] == part['father'].iloc[j]:
			neighbors_i.append(part['part_number'].iloc[j])
	

	




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


