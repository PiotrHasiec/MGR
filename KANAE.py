import kan
import numpy as np
from keras.datasets import mnist
from sklearn import datasets, manifold
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker
from sklearn.model_selection import train_test_split
import torch

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


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
n_samples = 6000
S_points, S_color = datasets.make_s_curve(n_samples,  random_state=0)
X_train, X_test, y_train, y_test =train_test_split(S_points,S_color)
model = kan.KAN(width=[3,2,3],grid=5,device="cuda",k=3,grid_range=[0,1],noise_scale=0.01,seed=2137)

dataset = {"train_input":torch.tensor(X_train),"train_label":torch.tensor(X_train),"test_input":torch.tensor(X_test),"test_label":torch.tensor(X_test)}
# model.cuda(0)
# model(dataset['train_input'])
# dataset['train_input'].detach().numpy()



# model.train(dataset , opt="Adam", steps=int(n_samples//4),lamb=0.1,lr=1,device="cuda",lamb_l1=2)

# model.train(dataset , steps=250,device="cuda",opt="Adam",lr=0.01)
# model.train(dataset , steps=100,device="cuda",opt="Adam",lr=0.001)
# model.train(dataset , steps=50,device="cuda",small_reg_factor=0,lamb_l1=0,lamb_entropy=0)
# model.train(dataset , steps=200,device="cuda",opt="Adam",small_reg_factor=0,lamb_l1=0,lamb_entropy=0,lr=0.1)
model.train(dataset , steps=100,device="cuda")
model.plot()
plt.show()





plot_3d(X_test, y_test, "Before reconstruction")

d2d = model(dataset["test_input"].to("cuda"))
plot_3d( d2d.cpu().detach().numpy(), y_test, "After reconstruction" )
plt.show()