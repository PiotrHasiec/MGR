from DensityMetricEstimator import DensityMEtricEstimator
from sklearn.neighbors import KNeighborsClassifier,KernelDensity
from keras.datasets import mnist

import numpy as np
from sklearn.metrics import DistanceMetric

from sklearn.decomposition import PCA



(x_train, y_train), (x_test, y_test) = mnist.load_data()

transformator = PCA(30)
x_train = np.reshape(x_train, (-1,28*28))
x_test = np.reshape(x_test, (-1,28*28))
# x_train = transformator.fit_transform(x_train)
x_train, y_train = x_train[:500], y_train[:500]
x_test, y_test =x_test[:120], y_test[:120]
# x_test = transformator.transform(x_test)

kde = KernelDensity(kernel="gaussian",bandwidth="scott").fit(x_train)
metric = DensityMEtricEstimator(kde=kde)





KNN_Classifier_Normal = KNeighborsClassifier(5)

KNN_Classifier_DensityMetric = KNeighborsClassifier(5,algorithm='ball_tree',metric=metric.calculate_dist)

KNN_Classifier_Normal.fit(x_train,y_train)
print(KNN_Classifier_Normal.score(x_test,y_test))

print("Fitting to density metric: ")
KNN_Classifier_DensityMetric.fit(x_train,y_train)
print("Scoring in density metric: ")
print(KNN_Classifier_DensityMetric.score(x_test,y_test))
