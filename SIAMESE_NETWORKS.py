from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import MaxPooling2D
import os
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker


    
# specify the shape of the inputs for our network
IMG_SHAPE = (28, 28, 1)
# specify the batch size and number of epochs
BATCH_SIZE = 128
EPOCHS = 2

MARGIN = 4
# define the path to the base output directory
BASE_OUTPUT = "output"
# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

def build_siamese_model(inputShape, embeddingDim=48):
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)
    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.25)(x)    
    # prepare the final outputs
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)
    # build the model
    model = Model(inputs, outputs)
    # return the model to the calling function
    return model

def plot_2d(points, points_color, title):
    x, y, = points.T
    plt.scatter(x,y, c= points_color,cmap="Spectral")
    plt.title(title)
    plt.show()
    
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
    
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np


def make_triplets(images, labels):

	triplesImages = []
	pairLabels = []

	numClasses = len(np.unique(labels))
	idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

	for idxA in range(len(images)):

		currentImage = images[idxA]
		label = labels[idxA]

		idxB = np.random.choice(idx[label])
		posImage = images[idxB]

		negIdx = np.where(labels != label)[0]
		negImage = images[np.random.choice(negIdx)]

		triplesImages.append([currentImage,posImage, negImage])
		pairLabels.append([1,1,0])

	return (np.array(triplesImages), np.array(pairLabels))

def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (anchor, positive, negative) = vectors
    # compute the sum of squared distances between the vectors
    ap_distance = K.sum(K.square(anchor - positive), axis=1,
        keepdims=True)
    an_distance = K.sum(K.square(anchor - negative), axis=1,
        keepdims=True)
    # return the euclidean distance between the vectors
    return tf.concat([K.sqrt(K.maximum(ap_distance, K.epsilon())),K.sqrt(K.maximum(an_distance, K.epsilon()))],1)

def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)
 


from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Lambda
import tensorflow as tf
from keras.datasets import fashion_mnist,mnist
import numpy as np


print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()

colors_ = [0,1,2,3,4,5,6,7,8,9]
c = [colors_[i] for i in testY ]

trainX = trainX / 255.0
testX = testX / 255.0
# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = make_triplets(trainX, trainY)
(pairTest, labelTest) = make_triplets(testX, testY)


# configure the siamese network
print("[INFO] building siamese network...")
imgA = Input(shape=IMG_SHAPE)
imgB = Input(shape=IMG_SHAPE)
imgC = Input(shape=IMG_SHAPE)
featureExtractor = build_siamese_model(IMG_SHAPE,embeddingDim=2)

points2scatter = featureExtractor(testX).numpy()


plot_2d(points2scatter,c, "f_mnist")


featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
featsC = featureExtractor(imgC)
                          
# finally, construct the siamese network
distance = Lambda(euclidean_distance)([featsA, featsB,featsC])

model = Model(inputs=[imgA, imgB,imgC], outputs=distance)


@tf.function
def triplet_loss(y_true, y_pred):
    ap_distance, an_distance = y_pred[:,0],y_pred[:,1]
    print(y_pred)
    # Computing the Triplet Loss by subtracting both distances and
    # making sure we don't get a negative value.
    loss = ap_distance - an_distance
    loss = K.maximum(loss + MARGIN, 0.0)
    return loss

# compile the model
print("[INFO] compiling model...")
import keras
model.compile(loss=triplet_loss, optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"])
# train the model
print("[INFO] training model...")
history = model.fit(
    [pairTrain[:, 0], pairTrain[:, 1],pairTrain[:, 2]], labelTrain[:],
    validation_data=([pairTest[:, 0], pairTest[:, 1], pairTest[:, 2]], labelTest[:]),
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS)


# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(MODEL_PATH)


points2scatter = featureExtractor(testX).numpy()


plot_2d(points2scatter,c, "f_mnist")




# plot the training history
print("[INFO] plotting training history...")
plot_training(history, PLOT_PATH)