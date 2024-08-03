from NDimSampling import PunktyOdleglosci
import tensorflow as tf
import keras 
from keras import Model
from keras.layers import Dropout, Dense, Input
import keras.backend as K
import numpy as np

MARGIN = 0.1
def build_siamese_model(inputShape, embeddingDim=48):
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)
    x = Dense(15,"tanh")(inputs)
    x = Dropout(0.25)(x)
    x = Dense(15,"tanh")(inputs)
    x = Dropout(0.25)(x)    
    x = Dense(5,"tanh")(inputs)
    
    # prepare the final outputs
  
    featureSpace = Dense(embeddingDim)(x)
    
    
    model = Model(inputs, featureSpace)
    # return the model to the calling function
    return model

@tf.function
def margin_mse(y_true, y_pred):

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


distancemodel = keras.models.Sequential()
from keras.constraints import non_neg
# distancemodel.add(keras.layers.Subtract())
distancemodel.add(Dense(25,"tanh",kernel_constraint=non_neg()))
distancemodel.add(Dropout(0.25))
distancemodel.add(Dense(25,"tanh",kernel_constraint=non_neg()))
distancemodel.add(Dropout(0.25))
distancemodel.add(Dense(3,"relu",kernel_constraint=non_neg()))
distancemodel.add(Dense(1,"relu",use_bias=False,kernel_constraint=non_neg()))

# distance = Lambda(euclidean_distance)([features1,features2])
distance  = distancemodel(tf.concat([features1,features2],-1))
model = Model(inputs=[Data1, Data2], outputs=distance)
model.compile("adam",margin_mse)
model.summary()
# pointGenerator = PunktyOdleglosci(DATASIZE,1000,-10,10,0.5*MARGIN)

# pointGenerator.generuj(1)
# points,dists = pointGenerator.make_pairs(4)

from DataGeneration import DataGenerator
sg = DataGenerator(1)
points = sg.generate_2d_internal_sphere(10)
pointsPair,dists = sg.distances(points)
c = [i for i in range( len(pointsPair))]
from matplotlib import pyplot as plt
points2model = pointsPair[:,0]
points2 = feature_extractor(points2model).numpy()
plt.scatter(points2.T[0],points2.T[1],c = c,cmap='hsv')
plt.show()

model.fit([pointsPair[:,0],pointsPair[:,1]],dists[:],3*16,40)




points2scatter0 = pointsPair[:,0]

(a,b) = (points2scatter0[:,0],points2scatter0[:,1])


plt.scatter(points2scatter0[:,0],points2scatter0[:,1],c = c,cmap='hsv')


points2model = pointsPair[:,0]

plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')



from sklearn.manifold import MDS

transformer  = MDS(dissimilarity="precomputed")

dissim = np.zeros((len(points),len(points)))
for i in range(len(points)):
    for j in range(len(points)):
        dissim[i][j] = sg.points2dist(points[i],points[j])
        
pointsMDS = transformer.fit_transform(dissim)
c = [i for i in range( len(pointsMDS))]


dissimMDS = np.zeros((len(pointsMDS),len(pointsMDS)))
for i in range(len(pointsMDS)):
    for j in range(len(pointsMDS)):
        dissimMDS [i][j] = np.linalg.norm(pointsMDS[i]-pointsMDS[j])
        
        

dissimNN = np.zeros((len(points),len(points)))
dataNN = np.reshape(model([pointsPair[:,0],pointsPair[:,1]]).numpy(), (len(points),len(points)))
for i in range(len(points)):
    for j in range(len(points)):
        dissimNN[i][j] = dataNN[i][j]
        
        
print(np.mean( (dissim-dissimMDS)**2))
print(np.mean( (dissim-dissimNN)**2))   
# print(transformer.)
points2 = feature_extractor(points).numpy()
plt.scatter(points2.T[0],points2.T[1],c = c,cmap='hsv')
plt.show()

plt.scatter(pointsMDS.T[0],pointsMDS.T[1],c = c,cmap='hsv')

plt.show()