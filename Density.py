import scipy.linalg
from sklearn.neighbors import KernelDensity,NearestNeighbors
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
# from keras.datasets import mnist
from sklearn import datasets, manifold
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker
from sklearn.model_selection import train_test_split
import scipy.integrate as integrate
# import keras
from scipy.spatial.distance import mahalanobis
# (x_train, y_train), (x_test, y_test)  = mnist.load_data()
# kde = KernelDensity(kernel='gaussian', bandwidth="silverman").fit(np.reshape(x_train,(-1,28*28)))
# print("Szacowanie")
# print(np.exp(kde.score_samples(np.reshape(x_test,(-1,28*28))[:100] )))

def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1000))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    
def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()
def dist(x,y,kde):
        normal_dist = np.linalg.norm(y-x)
        curve = lambda t: x + (y-x)*t
        t = np.linspace(0,1,100,endpoint=True)
        points =  [curve(ti) for ti in t]
        det = np.exp(-kde.score_samples(points)/2)
        det = np.where(abs(det) == np.inf, 1000,det)
        s = np.linalg.norm(points,axis=1)
        dist_ = integrate.trapezoid(det,dx = t[1]-t[0])*normal_dist
        if np.isnan(dist_):
            print(det)
        return(dist_)
        

def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    # ax.xaxis.set_major_formatter(ticker.NullFormatter())
    # ax.yaxis.set_major_formatter(ticker.NullFormatter())
    
# points,t = datasets.make_s_curve(3000,noise=0.04)
# points= np.array(points)
# kde = KernelDensity(kernel='epanechnikov', bandwidth="silverman").fit(points)
# print("Szacowanie")
# print(np.exp(kde.score_samples(points[:100])))

# probdens = np.exp(kde.score_samples(points))
# probdens =probdens/np.max(probdens)
# print(probdens[:100])
# colors = [(heat,0,0) for heat in  probdens ]

# plot_3d(points,colors,"heatmap")
# plt.show()

def v(t,J):
    return [J[0][0]*t[0]+J[0][1]*t[1],J[1][0]*t[0]+J[1][1]*t[1]]

def f(x):
    return np.array([np.sin(4*np.pi*x[0])*x[0],np.cos(4*np.pi*x[0])*x[0] ,x[1]] )

j=[[1,1],
   [1,7]]

N = 800
DIM = 2
x = np.random.uniform(0,1,(DIM,N))
# x = np.linspace(0,1,int(np.sqrt(N)),endpoint=True)
# X, Y = np.meshgrid(x, x)
# X = X.reshape((np.prod(X.shape),))
# Y = Y.reshape((np.prod(Y.shape),))
# x = np.stack([X,Y])
# np.random.shuffle(x)
# x= x.reshape(2,-1)
# t_x = np.array(f(x))[:2]
t_x = np.array(v(x,j))

print(  np.cov(x)  )
print(  1/np.linalg.det(np.cov(x))  )
print("\n")



plt.hist2d(t_x[0],t_x[1],30)
plt.show()
plt.scatter(t_x[0],t_x[1])
# plt.hist(t_x[0],30)
# plt.show()

# plt.hist(t_x[1],30)
# plt.show()

rand_i = np.random.randint(0,N-1)
rand_sample = t_x.T[rand_i].reshape(1, -1)

K =x.shape[1]
kde = KernelDensity(kernel="gaussian",bandwidth="silverman").fit(t_x.T)
print(np.linalg.det(np.matrix(j)))
det  =np.linalg.norm(np.cross(j[0], j[1]))
print(det)

density1 = np.exp(kde.score_samples(rand_sample))

neighbors_ = NearestNeighbors(n_neighbors=K).fit(t_x.T)

NN_dist,NN_ind = neighbors_.kneighbors(rand_sample)

_K_neighbors =t_x.T[NN_ind ][0] 
plt.scatter(_K_neighbors.T[0],_K_neighbors.T[1],c="g")
plt.scatter(rand_sample[0][0],rand_sample[0][1],c="r")
plt.show()
mean = np.repeat(np.average(_K_neighbors,0),K).reshape((2,-1))
_K_neighbors = (_K_neighbors.T - mean ).T
density2 = (K/(np.pi*np.max(NN_dist)**DIM))

print("x: ",np.std(_K_neighbors.T[0]) )
print("y: ",np.std(_K_neighbors.T[1]) )
print(density1,density2)
print(1/(density1),N/(density2))


matrix = np.cov(np.array(_K_neighbors).T) 
print("Macierz kowariancji sąsiadow:\n")
print(matrix)
print("Macierz kowariancji wszystkich danych: \n")
all_data_matrix =np.cov(np.array(t_x))
print(all_data_matrix)
print("\n")

print( np.sqrt(np.linalg.det(all_data_matrix)*(12**DIM)) )
print("\n")

print( np.sqrt(np.linalg.det(all_data_matrix) / np.linalg.det(np.cov(x))) )
print( np.linalg.det(np.linalg.inv(all_data_matrix)) )
print("\n")


d = np.var(_K_neighbors,0)
estimated_J = np.array([[d[0],0],[0,d[1]]])
print("Jakobian w bazie ortogonalnej: ",estimated_J)
print("Jego det: ", np.linalg.det(estimated_J))

estimated_J_norm = density1*estimated_J/np.linalg.det(estimated_J)
import scipy
decomposed = np.linalg.cholesky(all_data_matrix)
# decomposed = scipy.linalg.sqrtm(matrix)
# scaled = np.sqrt((det)/np.linalg.det(decomposed) )*decomposed
# print( scaled )
# print("\n")

print(rand_sample)
print(x.T[rand_i])
invJ = np.linalg.inv(np.array(j))
print( np.matmul(rand_sample,np.matmul(invJ,rand_sample.T)))

invScaled = np.linalg.pinv(estimated_J_norm)
print( np.matmul(rand_sample,np.matmul(invScaled,rand_sample.T)))

mds = manifold.MDS(n_components=2,metric=True,dissimilarity="precomputed",verbose=True)

NN_dist,NN_ind = neighbors_.kneighbors(x.T,n_neighbors=K)
dissimilarity_mat = np.ones((x.shape[1],x.shape[1]))
for i,xi in enumerate(x.T):
    
   
    for j in NN_ind[i]:
        # dissimilarity_mat[i][j] = NN_dist[i][j]
        dissimilarity_mat[i][j] = dist(x.T[i],x.T[j],kde)
        dissimilarity_mat[j][i] = dissimilarity_mat[i][j]
    
    
points2 = mds.fit_transform(dissimilarity_mat,)
# mds = manifold.MDS(n_components=2)
# points2 = mds.fit_transform(x.T)
colors_ = [(1.,0.,0.),(1.,1.,0.),(1.,0.,1.),(0.,1.,0.),(0.,1.,1.),(0.,0.,1.),(1.,1.,0.5),(1.,0.5,0.),(1.,0.,0.5),(0.5,1.,0.)]
y_color = [ ( i/N,(2*i)%N/N,(3*i)%N/N ) for i in range(x.shape[1]) ]
plot_2d(points2,y_color,"reconstructed")
plot_2d(x.T,y_color,"oryginal")
plt.show()

