from DensityMetricEstimator import DensityMEtricEstimator
from sklearn.neighbors import KNeighborsClassifier,KernelDensity
from keras.datasets import mnist


import numpy as np
from sklearn.metrics import DistanceMetric

from sklearn.decomposition import PCA



(x_train, y_train), (x_test, y_test) = mnist.load_data()

transformator = PCA(20)
x_train = np.reshape(x_train, (-1,28*28))
x_test = np.reshape(x_test, (-1,28*28))
x_train = transformator.fit_transform(x_train)
x_train, y_train = x_train[:5300], y_train[:5300]
x_test, y_test =x_test[:120], y_test[:120]
# x_test = transformator.transform(x_test)

kde = KernelDensity(kernel="gaussian",bandwidth="scott").fit(x_train)
metric = DensityMEtricEstimator(kde=kde)





KNN_Classifier_Normal = KNeighborsClassifier(5)

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import locally_linear_embedding, LocallyLinearEmbedding

# nbrs = NearestNeighbors(n_neighbors = 5,
#                         metric = 'precomputed')
# prec_tab_metric = metric.dist_tab(x_train)
# nbrs.fit(prec_tab_metric ) # NearestNeighbors instance with a precomputed distance matrix

# Z,_ =  locally_linear_embedding(nbrs,
#                                 n_neighbors = 5,
#                                 n_components = 2) # Passing down the NearestNeighbors instance with a precomputed distance matrix

LLE =LocallyLinearEmbedding()
Z=LLE.fit_transform(x_train)
import matplotlib.pyplot as plt

plt.scatter(Z.T[0],Z.T[1])
plt.show()