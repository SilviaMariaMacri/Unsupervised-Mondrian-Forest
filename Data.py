

data = {'X0':X[:,0],'X1':X[:,1],'X2':X[:,2],'y':y}

data=pd.DataFrame(data)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(np.array(data.query('y==0'))[:,0],np.array(data.query('y==0'))[:,1],np.array(data.query('y==0'))[:,2],alpha=0.5,color='b')
ax.scatter3D(np.array(data.query('y==1'))[:,0],np.array(data.query('y==1'))[:,1],np.array(data.query('y==1'))[:,2],alpha=0.5,color='orange')
#ax.scatter3D(np.array(data.query('y==2'))[:,0],np.array(data.query('y==2'))[:,1],np.array(data.query('y==2'))[:,2],alpha=0.5,color='g')


#%%

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
dat = datasets.make_circles(n_samples=200,noise=0.05,random_state=500,factor=0.5)
X = dat[0]
y = dat[1]
fig,ax=plt.subplots()
ax.scatter(X[:,0],X[:,1])
#%% make moons 2D
dat = datasets.make_moons(n_samples=200,noise=0.08,random_state=500)
X = dat[0]
y = dat[1]

#%% make moons 3D
dat = datasets.make_moons(n_samples=100,noise=0.07,random_state=0)
X = dat[0]
dim3 = np.random.normal(0, 0.2, len(X))
X = np.hstack((X[:,0].reshape((len(X),1)),X[:,1].reshape((len(X),1)),dim3.reshape((len(X),1))))
y = dat[1]






#%% makecircles 3D

dat1 = datasets.make_circles(n_samples=70,noise=0.1,random_state=500,factor=0.9)
X1 = dat1[0]


dat2 = datasets.make_circles(n_samples=70,noise=0.1,random_state=550,factor=0.9)
X2 = dat2[0]


cerchio1=np.hstack((X1[:,0].reshape((len(X1),1)),X1[:,1].reshape((len(X1),1))-1,np.random.normal(0,0.1,len(X1)).reshape((len(X1),1))))
cerchio2=np.hstack((np.random.normal(0,0.1,len(X2)).reshape((len(X2),1)),X2[:,0].reshape((len(X2),1)),X2[:,1].reshape((len(X2),1))))


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(cerchio1[:,0],cerchio1[:,1],cerchio1[:,2],alpha=0.5)
ax.scatter3D(cerchio2[:,0],cerchio2[:,1],cerchio2[:,2],alpha=0.5)

X = np.vstack([cerchio1,cerchio2])
y = np.hstack([np.zeros(len(cerchio1)),np.ones(len(cerchio2))])#.reshape(len(X1)+len(X2),)




#%% cilindro

rotation_axis = np.array([0, 0, 1])
angle = np.linspace(0,360,20)
Xinit=[1,1,0]
X = []
Xb = [0,0,0]

for i in range(len(angle)-1):
	rotation_degrees = angle[i]
	rotation_radians = np.radians(rotation_degrees)
	rotation_vector = rotation_radians * rotation_axis
	rotation = R.from_rotvec(rotation_vector)
	X_rot = rotation.apply(Xinit)
	X.append(list(X_rot))
Xinit = np.array(X) #+ np.random.normal(0,0.1,(len(X),3))
y =  np.zeros(len(Xinit))

for i in range(3):
	np.random.seed(0)
	Xt = np.array(Xinit) + np.random.normal(0,0.1,(len(Xinit),3)) +i*0.3#np.array([0,0,0.3])
	X = np.vstack([X,Xt])
	y = np.hstack([y,np.zeros(len(Xt))])
	np.random.seed(1)
	Xbt = np.array(Xb) + np.random.normal(0,0.1,(10,3)) +i*0.3#np.array([0,0,0.3])
	X = np.vstack([X,Xbt])
	y = np.hstack([y,np.ones(len(Xbt))])
	
	

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:,0],X[:,1],X[:,2],alpha=0.5)


#%% makemoons 3D


dat = datasets.make_moons(n_samples=20,noise=0.05,random_state=500,shuffle=False)
X = dat[0] #+ np.vstack([np.zeros(len(dat[0])),np.hstack([np.zeros(int(len(dat[0])/2)),-np.ones(int(len(dat[0])/2))])]).T
y = dat[1]
np.random.seed(500)
dim3 = np.random.normal(0, 0.01, len(X))#np.zeros(len(Xdat))#
X = np.hstack((X[:,0].reshape((len(X),1)),X[:,1].reshape((len(X),1)),dim3.reshape((len(X),1))))

for i in range(5):
	dat = datasets.make_moons(n_samples=20,noise=0.05,random_state=i,shuffle=False)
	Xdat = dat[0]
	ydat = dat[1]
	np.random.seed(i)
	dim3 = np.random.normal(0, 0.01, len(Xdat)) #np.zeros(len(Xdat))#
	Xdat = np.hstack((Xdat[:,0].reshape((len(Xdat),1)),Xdat[:,1].reshape((len(Xdat),1)),dim3.reshape((len(Xdat),1))))
	Xdat = Xdat +i*np.array([0,0.2,0.2])
	X = np.vstack([X,Xdat])
	y = np.hstack([y,ydat])	
	

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:,0],X[:,1],X[:,2],alpha=0.5)




#%% semisfere


rotation_axis = np.array([0, 0, 1])
angle = np.linspace(0,180,10)
Xinit=[1,0,0]
X = []
Xb = [0,0,0]

for i in range(len(angle)):
	rotation_degrees = angle[i]
	rotation_radians = np.radians(rotation_degrees)
	rotation_vector = rotation_radians * rotation_axis
	rotation = R.from_rotvec(rotation_vector)
	X_rot = rotation.apply(Xinit)
	X.append(list(X_rot))
X = np.array(X) #+ np.random.normal(0,0.1,(len(X),3))
Xinit = X.copy()

for i in range(len(angle)):
	rotation_axis = np.array([1, 0, 0])
	rotation_degrees = angle[i]
	rotation_radians = np.radians(rotation_degrees)
	rotation_vector = rotation_radians * rotation_axis
	rotation = R.from_rotvec(rotation_vector)
	X_rot = rotation.apply(Xinit + np.random.normal(0,0.05,(len(Xinit),3)))
	X = np.vstack([X,X_rot])
y = np.zeros(len(X))
	
blob = datasets.make_blobs(n_samples=30,n_features=3,centers=[(0,0,0.4)],cluster_std=0.1,random_state=18)
Xb = blob[0]
yb = np.ones(len(Xb))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:,0],X[:,1],X[:,2],alpha=0.5)
ax.scatter3D(Xb[:,0],Xb[:,1],Xb[:,2],alpha=0.5)


X = np.vstack([X,Xb])
y = np.hstack([y,yb])
#%%3D makemmons+blob

from scipy.spatial.transform import Rotation as R
rotation_axis = np.array([0, 1, 0])
angle = np.linspace(0,360,50)
n_samples=20
dat = datasets.make_moons(n_samples=n_samples,noise=0.05,random_state=len(angle)
						  ,shuffle=False)
len_dat = int(len(dat[0])/2)
X = dat[0][0:len_dat]
y = dat[1][0:len_dat]
X=np.hstack((X[:,0].reshape((len(X),1)),X[:,1].reshape((len(X),1)),np.zeros(len(X)).reshape((len(X),1))))

for i in range(len(angle)):
	rotation_degrees = angle[i]
	rotation_radians = np.radians(rotation_degrees)
	rotation_vector = rotation_radians * rotation_axis
	rotation = R.from_rotvec(rotation_vector)
	#if i%2==0:
	#	n_samples_i=20
	#else:
	n_samples_i=n_samples
	dat = datasets.make_moons(n_samples=n_samples_i,noise=0.05,random_state=i
						   ,shuffle=False)
	X_rot = dat[0][0:len_dat]
	y_rot = dat[1][0:len_dat]
	X_rot=np.hstack((X_rot[:,0].reshape((len(X_rot),1)),X_rot[:,1].reshape((len(X_rot),1)),np.zeros(len(X_rot)).reshape((len(X_rot),1))))

	X_rot = rotation.apply(X_rot)
	X = np.vstack([X,X_rot])

	#fig = plt.figure()
	#ax = plt.axes(projection='3d')
	#ax.scatter3D(X_rot[:,0],X_rot[:,1],X_rot[:,2],alpha=0.5)
	y = np.hstack([y,y_rot])#.reshape(len(y)+len(y_rot),)

df_X =pd.DataFrame(X)
X = np.array(df_X[(df_X[1]>0.2) & (df_X[1]<0.8)])
y = np.zeros(len(X))

blob = datasets.make_blobs(n_samples=30,n_features=3,centers=[(0,0.4,0)],cluster_std=0.1,random_state=18)
Xb = blob[0]
yb = np.ones(len(Xb))


'''
circle = datasets.make_circles(n_samples=70,noise=0.05,random_state=550,factor=.99)
X_circle = circle[0]
y_circle = circle[1]


cerchio=np.hstack((X_circle[:,0].reshape((len(X_circle),1)),np.random.normal(-0.2,0.05,len(X_circle)).reshape((len(X_circle),1))				   ,X_circle[:,1].reshape((len(X_circle),1))))
'''

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:,0],X[:,1],X[:,2],alpha=0.5)
ax.scatter3D(Xb[:,0],Xb[:,1],Xb[:,2],alpha=0.5)
#ax.scatter3D(cerchio[:,0],cerchio[:,1],cerchio[:,2],alpha=0.5)

X = np.vstack([X,Xb])
y = np.hstack([y,yb])







#%%  makemoons3D
from scipy.spatial.transform import Rotation as R
rotation_axis = np.array([0, 1, 0])
angle = np.linspace(0,90,10)
n_samples=20
dat = datasets.make_moons(n_samples=n_samples,noise=0,random_state=len(angle)
						  ,shuffle=False)
X = dat[0]
y = dat[1]
X=np.hstack((X[:,0].reshape((len(X),1))-0.5,X[:,1].reshape((len(X),1)),np.zeros(len(X)).reshape((len(X),1))))

for i in range(len(angle)):
	rotation_degrees = angle[i]
	rotation_radians = np.radians(rotation_degrees)
	rotation_vector = rotation_radians * rotation_axis
	rotation = R.from_rotvec(rotation_vector)
	#if i%2==0:
	#	n_samples_i=20
	#else:
	n_samples_i=n_samples
	dat = datasets.make_moons(n_samples=n_samples_i,noise=0,random_state=i
						   ,shuffle=False)
	X_rot = dat[0]
	y_rot = dat[1]
	X_rot=np.hstack((X_rot[:,0].reshape((len(X_rot),1))-0.5,X_rot[:,1].reshape((len(X_rot),1)),np.zeros(len(X_rot)).reshape((len(X_rot),1))))

	X_rot = rotation.apply(X_rot)
	X = np.vstack([X,X_rot])
	y = np.hstack([y,y_rot])#.reshape(len(y)+len(y_rot),)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:,0],X[:,1],X[:,2],alpha=0.5)
#ax.scatter3D(Xb[:,0],Xb[:,1],Xb[:,2],alpha=0.5)
#ax.scatter3D(cerchio[:,0],cerchio[:,1],cerchio[:,2],alpha=0.5)

#X = np.vstack([X,Xb])
#y = np.hstack([y,yb])
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
#%%
# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=170)
X=varied[0]
fig,ax=plt.subplots()
ax.scatter(X[:,0],X[:,1])
