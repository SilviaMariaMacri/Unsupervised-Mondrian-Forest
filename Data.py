# make circles in 3D
#make_s_curve(n_samples=100, *, noise=0.0, random_state=None)[source]


import numpy as np
import pandas as pd
from sklearn import datasets

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

#%% make blobs 3D
dat = datasets.make_blobs(n_samples=[50,50,30,30],n_features=2,cluster_std=0.9,random_state=18)
X = dat[0]
y = dat[1]
fig,ax = plt.subplots()
ax.scatter(X[:,0],X[:,1])


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:,0],X[:,1],X[:,2],alpha=0.5)

#%% make circles 2D
dat = datasets.make_circles(n_samples=100,noise=0.05,random_state=170,factor=0.5)
X = dat[0]
y = dat[1]

#%% make moons 2D
dat = datasets.make_moons(n_samples=100,noise=0.08,random_state=350)
X = dat[0]
y = dat[1]
plt.scatter(X[:,0],X[:,1])
#%% make moons 3D
dat = datasets.make_moons(n_samples=100,noise=0.07,random_state=0)
X = dat[0]
dim3 = np.random.normal(0, 0.2, len(X))
X = np.hstack((X[:,0].reshape((len(X),1)),X[:,1].reshape((len(X),1)),dim3.reshape((len(X),1))))
y = dat[1]

#%% iris 4D
iris = datasets.load_iris() 
X = iris.data
y = iris.target

data = pd.DataFrame(X)
data = data.drop_duplicates()
X = np.array(data)

y = list(y)
y.remove(2)
y=np.array(y)



#%% wine 

wine = datasets.load_wine()
X = wine.data
y= wine.target

#%% dati ruotati

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




#%% gaussiane
		
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

'''
mean5 = (2.5,-2.5)
cov5 = [[0.01,b1+0.1], [b1+0.1,0.01]]
np.random.seed(50)
x5 = np.random.multivariate_normal(mean5, cov5, a)
'''



			

X = np.vstack([x1,x2,x3,x4])
#class
y1 = np.zeros(a)
y2 = np.ones(a)
y3 = 2*np.ones(a)
y4 = 3*np.ones(a)
y = np.hstack([y1,y2,y3,y4])


import pylab as plt			
fig,ax = plt.subplots()
#ax.scatter(x1[:,0],x1[:,1],color='b')
#ax.scatter(x2[:,0],x2[:,1],color='b')
ax.scatter(X[:,0],X[:,1])




X_originale = X.copy()





#%%  gaussiane verticali

b=0.02
a=50
mean1 = (0, 0)
cov1 = [[b,0], [0,b]]
np.random.seed(0)
x1 = np.random.multivariate_normal(mean1, cov1, a)

mean2 = (1,0)
cov2 = [[b,0], [0,b]]
np.random.seed(1)
x2 = np.random.multivariate_normal(mean2, cov2, a)


import pylab as plt
X = np.vstack([x1,x2])
fig,ax = plt.subplots()
ax.scatter(X[:,0],X[:,1])

#3D
dim3_1 = np.random.normal(0, 1, len(x1))
dim3_2 = np.random.normal(0, 1, len(x2))
X1 = np.hstack((x1[:,0].reshape((len(x1),1)),x1[:,1].reshape((len(x1),1)),dim3_1.reshape((len(x1),1))))
X2 = np.hstack((x2[:,0].reshape((len(x2),1)),x2[:,1].reshape((len(x2),1)),dim3_2.reshape((len(x2),1))))
X = np.vstack([X1,X2])


















#%% copiato da https://scikit-learn.org/stable/auto_examples/cluster/plot_linkage_comparison.html#sphx-glr-auto-examples-cluster-plot-linkage-comparison-py

np.random.seed(0)

n_samples = 150
#noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
#                                      noise=.05)
#noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
#blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
#no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)


