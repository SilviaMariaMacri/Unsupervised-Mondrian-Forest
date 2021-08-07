# cd "C:\Users\silvi\Desktop\Fisica\TESI\Tesi"


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
	


#%% Polytope dimensione generica + classificazione 

t0 = 0
lifetime = 5
#dist_matrix = DistanceMatrix(X)
number_of_iterations = 16
name = 'makeblobs_3D_'

for i in range(1,number_of_iterations):
	m_i,part_i = Mondrian(X,t0,lifetime,dist_matrix)
	namefile = name+str(i+1)
	SaveMondrianOutput(namefile,part_i,m_i)
	#PlotPolygon(X,part_i)
	
	part = json.load(open(namefile+'_part.json','r'))
	part = pd.DataFrame(part)
	m = json.load(open(namefile+'_m.json','r'))
	tagli_paralleli = False #True#,False
	score = 'min' #'var','centroid'
	weight = 'diff_min' #'var_ratio','ratio_centroid','diff_centroid',
	Classification(part,m,X,namefile,score,weight,tagli_paralleli)
	



#%% leggo file .json


number_of_iterations = 16
name = 'makemoons_2_'

list_part = []
list_m = []
list_class = []
list_conn_comp = []

for i in range(number_of_iterations):
	namefile = name+str(i+1)
	
	part = json.load(open(namefile+'_part.json','r'))
	part = pd.DataFrame(part)
	m = json.load(open(namefile+'_m.json','r'))
	classified_data = json.load(open(namefile+'_list_class.json','r'))
	conn_comp = json.load(open(namefile+'_conn_comp.json','r'))
	
	list_part.append(part)
	list_m.append(m)	
	list_class.append(classified_data)
	list_conn_comp.append(conn_comp)
	


#%% grafici

# grafico compatibilità classificazioni
coeff_medio = ClassificationScore(list_class)

number_of_iterations = 16
for i in range(number_of_iterations):

	i = 8	
	part = list_part[i]
	m = list_m[i]
	classified_data = list_class[i]
	conn_comp = list_conn_comp[i]
	
	# puoi fissare number_of_clusters
	number_of_clusters = 2
	#for number_of_clusters in range(len(conn_comp)):
	#PlotClass_2D(X,part,conn_comp,number_of_clusters)
	PlotClass_3D(X,part,conn_comp,number_of_clusters)








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
