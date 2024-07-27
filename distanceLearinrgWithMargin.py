from NDimSampling import PunktyOdleglosci
import tensorflow as tf
import keras 
from keras import Model
from keras.layers import Dropout, Dense, Input
import keras.backend as K


MARGIN = 0.5
def build_siamese_model(inputShape, embeddingDim=48):
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)
    x = Dense(10,"relu")(inputs)
    x = Dropout(0.25)(x)
    x = Dense(10,"relu")(inputs)
    x = Dropout(0.25)(x)    
    x = Dense(10,"relu")(inputs)
    x = Dropout(0.25)(x)  
    # prepare the final outputs
  
    featureSpace = Dense(embeddingDim)(x)
    
    
    model = Model(inputs, featureSpace)
    # return the model to the calling function
    return model

@tf.function
def triplet_loss(y_true, y_pred):

    loss = tf.reduce_mean( K.maximum(tf.abs(y_true- y_pred) - MARGIN,0))
    return loss

DATASIZE = 2
feature_extractor = build_siamese_model((DATASIZE),2)

Data1 = Input((DATASIZE))
Data2 = Input((DATASIZE))

features1 = feature_extractor(Data1)
features2 = feature_extractor(Data2)


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
        keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


from keras.layers import Lambda
# distance = Lambda(euclidean_distance)([features1,features2])

distancemodel = keras.models.Sequential()

distancemodel.add(Dense(10,"relu"))
distancemodel.add(Dropout(0.25))
distancemodel.add(Dense(10,"relu"))
distancemodel.add(Dropout(0.25))
# distancemodel.add(Dense(3,"relu"))
distancemodel.add(Dense(1,"relu"))

distance  = distancemodel(tf.concat([features1,features2],-1))
model = Model(inputs=[Data1, Data2], outputs=distance)
model.compile("adam",triplet_loss)
model.summary()
pointGenerator = PunktyOdleglosci(DATASIZE,1000,-10,10,MARGIN)

pointGenerator.generuj(0.4)
points,dists = pointGenerator.make_pairs(4)

model.fit([points[:,0],points[:,1]],dists[:],16,100)
from matplotlib import pyplot as plt
points2scatter1 = points[:,0]
plt.scatter(points2scatter1[:,0],points2scatter1[:,1])


points2model = points[:,0]
points2 = feature_extractor(points2model).numpy()

plt.scatter(points2.T[0],points2.T[1])
plt.show()




