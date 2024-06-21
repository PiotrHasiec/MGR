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
from sklearn.model_selection import train_test_split
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn import datasets, manifold

# specify the shape of the inputs for our network
DATA_SHAPE = (3)
# specify the batch size and number of epochs
BATCH_SIZE =8
EPOCHS = 30
KN = 500

# define the path to the base output directory
BASE_OUTPUT = "output"
# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

# def build_siamese_model(inputShape, embeddingDim=48):
#     # specify the inputs for the feature extractor network
#     inputs = Input(inputShape)
#     # define the first set of CONV => RELU => POOL => DROPOUT layers
#     x = Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(0.3)(x)
#     # second set of CONV => RELU => POOL => DROPOUT layers
#     x = Conv2D(64, (7, 7), padding="same", activation="relu")(x)
#     x = MaxPooling2D(pool_size=2)(x)
#     x = Dropout(0.3)(x)    
#     # prepare the final outputs
#     pooledOutput = GlobalAveragePooling2D()(x)
#     outputs = Dense(embeddingDim)(pooledOutput)
#     # build the model
#     model = Model(inputs, outputs)
#     # return the model to the calling function
#     return model

def build_siamese_model(inputShape, embeddingDim=48):
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)
    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Dense(30, activation="relu")(inputs)
    x = Dropout(0.3)(x)
    x = Dense(30, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(embeddingDim)(x)
    # build the model
    model = Model(inputs, outputs)
    # return the model to the calling function
    return model

import scipy.integrate as integrate
def calculate_dist(x,y,kde):
        normal_dist = np.linalg.norm(y-x)
        curve = lambda t: x + (y-x)*t
        t = np.linspace(0,1,20,endpoint=True)
        points =  [curve(ti) for ti in t]
        # det = np.power(np.exp(kde.score_samples(points)), )
        det = np.exp(-np.exp(kde.score_samples(points)*(1./float(x.shape[0]))))     
        # det = np.where(abs(det) == np.inf, 1000,det)

        normal_dist = integrate.trapezoid(det,dx = t[1]-t[0])*normal_dist
        return normal_dist
        return(dist_)

from tqdm import tqdm 

def make_pairs(images,kde,neighbors_):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
    images_flat = np.reshape(images,(-1,DATA_SHAPE))
    
    
    NN_dist,NN_ind = neighbors_.kneighbors(images_flat,n_neighbors=KN)
    
    dists = []
    pairsImages =[]
    for i,xi in tqdm(enumerate(images_flat)):
        for j in NN_ind[i]:
            pairsImages.append([images[0],images[j]])
            dists.append(calculate_dist(images_flat[0],images_flat[j],kde))

    return (np.array(pairsImages), np.array(dists))

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
S_points, S_color = datasets.make_swiss_roll(n_samples,noise=0., random_state=0)
X_train, X_test, y_train, y_test = train_test_split(S_points,S_color,)
# (trainX, trainY), (testX, testY) = mnist.load_data()
trainX =X_train
testX = X_test
colors_ = [(1.,0.,0.),(1.,1.,0.),(1.,0.,1.),(0.,1.,0.),(0.,1.,1.),(0.,0.,1.),(1.,1.,0.5),(1.,0.5,0.),(1.,0.,0.5),(0.5,1.,0.)]

Data1 = Input(shape=DATA_SHAPE)
Data2 = Input(shape=DATA_SHAPE)
featureExtractor = build_siamese_model(DATA_SHAPE,embeddingDim=2)

points2scatter = featureExtractor(trainX).numpy()

plt.scatter(points2scatter.T[0],points2scatter.T[1],c = y_train,alpha=0.6)
plt.show()
# for i in range (0,10):
#     points2scatter_i = points2scatter[testY ==i]
#     plt.scatter(points2scatter_i.T[0],points2scatter_i.T[1],c = colors_[i], label = i,alpha=0.6)
# plt.legend()
# plt.show()

featsA = featureExtractor(Data1)
featsB = featureExtractor(Data2)
distance = Lambda(euclidean_distance)([featsA, featsB])
model = Model(inputs=[Data1, Data2], outputs=distance)


kde = KernelDensity(kernel="epanechnikov",bandwidth="silverman").fit(trainX)
neighbors_ = NearestNeighbors(n_neighbors=KN).fit(trainX)
(pairTrain, labelTrain) = make_pairs(trainX,kde,neighbors_)
# (pairTest, labelTest) = make_pairs(testX,kde,neighbors_)


import keras
# compile the model
print("[INFO] compiling model...")
model.compile(loss=keras.losses.MSE, optimizer="adam",
    metrics=["mse"])
# train the model
print("[INFO] training model...")
history = model.fit(
    [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
   
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS)

# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(MODEL_PATH)


points2scatter = featureExtractor(trainX).numpy()


plt.scatter(points2scatter.T[0],points2scatter.T[1],c = y_train,alpha=0.6)
plt.show()

