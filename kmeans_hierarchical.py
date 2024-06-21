from sklearn.cluster import k_means,KMeans
from keras.datasets import mnist,fashion_mnist
import numpy as np
from sklearn.manifold import MDS
from sklearn import datasets
from sklearn.model_selection import train_test_split
n_samples = 2500

colors_ = [(1.,0.,0.),(1.,1.,0.),(1.,0.,1.),(0.,1.,0.),(0.,1.,1.),(0.,0.,1.),(1.,1.,0.5),(1.,0.5,0.),(1.,0.,0.5),(0.5,1.,0.)]
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (-1,28*28))[:n_samples]
x_test = np.reshape(x_test, (-1,28*28))
y_train = y_train[:n_samples]
c = [colors_[i] for i in y_train ]


# S_points, S_color = datasets.make_swiss_roll(n_samples,noise=0., random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(S_points,S_color)

# y_train = y_train[:n_samples]

# x_train = X_train

# cl1 = KMeans(100)
# cl1.fit(x_train)
# cl1.
c1_num  =1
c2_num = 1
centroid,label, interia = k_means(x_train,c1_num)
stds = []
for i in range(c1_num):
    stds.append(np.std(x_train[label==i]))
dissimilarity_matrix = np.zeros((x_train.shape[0],x_train.shape[0]))


centroid_2,label_2, interia_2 = k_means(centroid,c2_num )
stds2=[]
for i in range(c2_num ):
    stds2.append(np.std(centroid[label_2==i]))
std = np.std(centroid_2)
dissimilarity_matrix = np.zeros((x_train.shape[0],x_train.shape[0]))

for i,x1 in enumerate(x_train):
    for j,x2 in enumerate(x_train):
        if label[i]==label[j]:
            dissimilarity_matrix[i][j] = np.linalg.norm(x1-x2)/stds[label[i]]
        
        # else:    
        elif label_2[label[i]] == label_2[label[j]]:
            dissimilarity_matrix[i][j] = np.linalg.norm(centroid[label[j]]-centroid[label[i]])/stds2[label_2[label[i]]] +np.linalg.norm(x1-centroid[label[i]])/stds[label[i]] + np.linalg.norm(x2-centroid[label[j]])/stds[label[j]]
        else:
            dissimilarity_matrix[i][j] = np.linalg.norm(centroid_2[label_2[label[i]] ]-centroid_2[label_2[label[j]] ])/std + np.linalg.norm(x1-centroid[label[i]])/stds[label[i]] + np.linalg.norm(x2-centroid[label[j]])/stds[label[j]] + np.linalg.norm(centroid[label[j]]-centroid_2[label_2[label[j]] ])/stds2[label_2[label[j]]]  + np.linalg.norm(centroid[label[i]]-centroid_2[label_2[label[i]] ])/stds2[label_2[label[i]]] 
            
        dissimilarity_matrix[j][i] = dissimilarity_matrix[i][j]
        if i==j:
            continue






transformator = MDS(dissimilarity="precomputed",n_jobs=8,max_iter=200,verbose=1)
from sklearn.decomposition import PCA
pca=PCA(2)
# init = pca.fit_transform(x_train)
points2sctatter = transformator.fit_transform(dissimilarity_matrix)

import matplotlib.pyplot as plt

plt.scatter(points2sctatter.T[0],points2sctatter.T[1], c=c)
plt.show()