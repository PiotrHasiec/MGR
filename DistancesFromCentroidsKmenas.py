import numpy as np
from keras.datasets import mnist
from sklearn.cluster import kmeans_plusplus,k_means,KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, manifold
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker
from sklearn import manifold

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
    plt.show()
def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
# n_samples = 1500
# S_points, S_color = datasets.make_swiss_roll(n_samples,noise=0.3, hole=True, random_state=0)
# plot_3d(S_points, S_color, "Original S-curve samples")
# clusterizator  = KMeans(n_clusters=100)
# clusterizator.fit(S_points)    
# plot_3d(clusterizator.cluster_centers_,"Red","Centroids")

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# points = np.reshape(x_train[:15000],(-1,28*28))
# colors_ = [(1.,0.,0.),(1.,1.,0.),(1.,0.,1.),(0.,1.,0.),(0.,1.,1.),(0.,0.,1.),(1.,1.,0.5),(1.,0.5,0.),(1.,0.,0.5),(0.5,1.,0.)]
# y_train_color = [colors_[i] for i in y_train[:15000]]
# for i in [128]:
#     clusterizator  = KMeans(n_clusters=i)
#     points = clusterizator.fit_transform(points)

# iso = manifold.Isomap(n_neighbors=5,n_components=2)
# points = iso.fit_transform(points)
# plot_2d(points,y_train_color,"Points in 2D")




