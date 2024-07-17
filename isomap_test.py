import sklearn.manifold as manifold
from keras.datasets import mnist
from sklearn.decomposition import PCA
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()

transformator = PCA(30)
x_train = np.reshape(x_train, (-1,28*28))
x_test = np.reshape(x_test, (-1,28*28))
# x_train = transformator.fit_transform(x_train)
x_train, y_train = x_train[:20000], y_train[:20000]
# x_test, y_test =x_test[:100], y_test[:100]

isomap = manifold.Isomap(n_neighbors=3)
isomap.fit(x_train)
points2scatter = isomap.transform(x_test)

colors_ = [(1.,0.,0.),(1.,1.,0.),(1.,0.,1.),(0.,1.,0.),(0.,1.,1.),(0.,0.,1.),(1.,1.,0.5),(1.,0.5,0.),(1.,0.,0.5),(0.5,1.,0.)]

c = [colors_[i] for i in y_test ]

import matplotlib.pyplot as plt

plt.scatter(points2scatter.T[0],points2scatter.T[1],c=c)
plt.show()