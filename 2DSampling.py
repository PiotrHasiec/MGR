import numpy as np
from scipy.integrate import trapezoid
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker
from sklearn.model_selection import train_test_split
import math


def plot_3d(points, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1000))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()
def f(x):
    return np.array([np.sin(x[0])*x[0],np.cos(x[0])*x[0],x[1]])

x = np.random.uniform(0,20*np.pi,(2,100000))
print(np.max(x))
p = f(x)
# p = np.append(x.T,p,axis=1)
plt.scatter(p[0],p[1])
plt.show()
plt.hist(p[0],100)
# plt.hist2d(p[0],p[1],100)
plt.show()

x = np.log(np.random.uniform(1,100,(2,100000)))*(20*np.pi/np.log(100))
print(np.max(x))
p = f(x)
# p = np.append(x.T,p,axis=1)
plt.scatter(p[0],p[1])

plt.show()
plt.hist(p[0],100)
# plt.hist2d(p[0],p[1],100)
plt.show()