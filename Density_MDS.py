from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.datasets import mnist
from sklearn.neighbors import KernelDensity,NearestNeighbors
import os
import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn import datasets, manifold
from scipy.sparse.csgraph import dijkstra,floyd_warshall,shortest_path
from scipy.sparse import csr_matrix
# specify the shape of the inputs for our network
DATA_SHAPE = (30)
# specify the batch size and number of epochs
BATCH_SIZE =128
EPOCHS = 10
KN = 30


# define the path to the base output directory
BASE_OUTPUT = "output"
# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
F = 1./DATA_SHAPE

import scipy.integrate as integrate
def calculate_dist(x,y,kde):
        dist_ = np.linalg.norm(y-x)
        curve = lambda t: x + (y-x)*t
        t = np.linspace(0,1,40,endpoint=True)
        points =  [curve(ti) for ti in t]
        det = np.exp(-np.exp(kde.score_samples(points)*F))
        det[abs(det) == np.inf] = 10000 
        dist_ = integrate.trapezoid(det,t)*dist_
        # if np.isnan(dist_):
        #     print(det)
        # return(normal_dist)
        return(dist_)

from tqdm import tqdm 

def make_pairs(images,kde,neighbors_):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
    images_flat = np.reshape(images,(-1,DATA_SHAPE))
    
    
    NN_dist,NN_ind = neighbors_.kneighbors(images_flat,n_neighbors=KN)
    
    dissimilarity_mat = np.ones((images_flat.shape[0],images_flat.shape[0]))*np.inf
    for i,xi in tqdm(enumerate(images_flat)):
        
        
        for j in NN_ind[i]:

            dissimilarity_mat[i][j] = calculate_dist(images_flat[i],images_flat[j],kde)
            dissimilarity_mat[j][i] = dissimilarity_mat[i][j]

    return dissimilarity_mat

def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
        keepdims=True)
    # return the euclidean distance between the vectors
    return (K.maximum(sumSquared, K.epsilon()))


print("[INFO] dataset...")
n_samples =1000
(trainX, trainY), (testX, testY) = mnist.load_data()
from sklearn.decomposition import PCA

# X_train, X_test, y_train, y_test = train_test_split(S_points,S_color,)
# (trainX, trainY), (testX, testY) = mnist.load_data()
# trainX =X_train
# testX = X_test
colors_ = [(1.,0.,0.),(1.,1.,0.),(1.,0.,1.),(0.,1.,0.),(0.,1.,1.),(0.,0.,1.),(1.,1.,0.5),(1.,0.5,0.),(1.,0.,0.5),(0.5,1.,0.)]

c = [colors_[i] for i in trainY ]

trainX = np.reshape(trainX,(-1,28*28))[:n_samples]
trainX = PCA(n_components=DATA_SHAPE).fit_transform(trainX)

kde = KernelDensity(kernel="epanechnikov",bandwidth="scott").fit(trainX)
neighbors_ = NearestNeighbors(n_neighbors=KN).fit(trainX)
diss_mat = make_pairs(trainX,kde,neighbors_)
# (pairTest, labelTest) = make_pairs(testX,kde,neighbors_)


graph = csr_matrix(diss_mat)

print("shortest")
dist_matrix, predecessors = shortest_path(csgraph=diss_mat, return_predecessors=True,directed=True)

dist_matrix22 = np.where(abs(dist_matrix) == np.inf, 0,dist_matrix)
diss_mat2 = np.where(abs(dist_matrix) == np.inf, 2*np.max(dist_matrix22),dist_matrix)
mds = manifold.MDS(n_components=2,metric=True,dissimilarity="precomputed",verbose=True,)
# isomap = manifold.Isomap(n_neighbors=KN)
# points2scatter = isomap.fit_transform(trainX)
# print("mds")
points2scatter = mds.fit_transform(diss_mat2)

plt.scatter(points2scatter.T[0],points2scatter.T[1],c = c[:n_samples],alpha=0.6)
plt.show()
