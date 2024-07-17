import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns


# sns.set(context="paper", style="white")

mnist = fetch_openml("mnist_784", version=1)

reducer = umap.UMAP(n_neighbors = 3)
data = mnist.data
embedding = reducer.fit_transform(data)

import numpy as np
colors_ = [(1.,0.,0.),(1.,1.,0.),(1.,0.,1.),(0.,1.,0.),(0.,1.,1.),(0.,0.,1.),(1.,1.,0.5),(1.,0.5,0.),(1.,0.,0.5),(0.5,1.,0.)]
c = np.array([ colors_[int(i)] for i in mnist.target])
# fig, ax = plt.subplots(figsize=(12, 10))
color = mnist.target.astype(int)
for i in range (0,10):
    inds = list(mnist.target ==str(i))
    points2scatter_i = embedding[inds]
    cc = c[mnist.target ==str(i)]
    plt.scatter(points2scatter_i[:,0],points2scatter_i[:,1],c = cc, label = i,alpha=0.6)
plt.legend()
plt.show()

plt.title("UMAP", fontsize=18)

plt.show()