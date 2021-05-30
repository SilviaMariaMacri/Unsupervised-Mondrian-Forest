# cd "C:\Users\silvi\Desktop\Fisica\TESI\Tesi"

import numpy as np
import random
from numpy.random import choice
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pyplot import cm


from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



import prova2


#%% data

 
dat = datasets.make_moons(n_samples=100,noise=0.1)
iris = datasets.load_iris()
'''
#iris
data = pd.DataFrame(iris.data)
data[[0,1,2]]

'''



#%% 2D

#make_moons
X = dat[0]
y = dat[1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)




#%% 3D


#make_moons
data = pd.DataFrame(dat[0])

altra_dim = np.random.normal(0, 1, len(data))
data[2] = altra_dim
data['class']=dat[1]


X = np.array(data[[0,1,2]])
y = dat[1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)



#%%  

t0=0
lifetime=2

part = Mondrian_completo_unsupervised(X,t0,lifetime)
#part_with_counts = Count(X,y,part)
#accuracy,cl = Class(X,y,part_with_counts)

PartitionPlot(X,y,part)


# part[['time', 'father', 'part_number', 'dim', 'distance', 'x', 'leaf']]

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








