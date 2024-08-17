
from typing import Any
import numpy as np
import keras
from tqdm import tqdm
import tensorflow_probability as tfp
import tensorflow as tf
from EvolutionCurve import EvolutionaryCurve,GradientDescentCurve
from DataGeneration import DataGenerator
from keras.src.saving import serialization_lib
serialization_lib.enable_unsafe_deserialization()
N = 30
def transpose(w):
    return  tf.einsum("abcd->abdc",w)

class FillTriangular(keras.Layer):
    def call(self, x):
        return tfp.math.fill_triangular(x)

class ABS(keras.Layer):
    def call(self, x):
        return tf.abs(x)
    
class MULTIPLY(keras.Layer):
    def __init__(self,inputSize, *, activity_regularizer=None, trainable=True, dtype=None, autocast=True, name=None, **kwargs):
        self.inputSize = inputSize
        super().__init__(activity_regularizer=activity_regularizer, trainable=trainable, dtype=dtype, autocast=autocast, name=name, **kwargs)
    def call(self, x):
        return tf.math.multiply(x,tf.eye(self.inputSize,dtype="float32"))
    
class tensorNetwork():
    def __init__(self,inputSize,structure,pathfinder:EvolutionaryCurve) -> None:
        # self.nn =keras.models.Sequential()
        input = keras.layers.Input((None,inputSize))
        x = keras.layers.Dense(structure[0],"relu")(input)
        for layer in structure[0:]:
            
            x = keras.layers.Dropout(0.2)(x)
            x = keras.layers.Dense(layer,"relu")(x)
        # x = keras.layers.Dropout(0.2)(x)    

        # tensorElements = keras.layers.Dense( inputSize**2 ,"relu",dtype='float32')(x)
        # tensor = keras.layers.Reshape((-1,inputSize,inputSize))(tensorElements)
        
        tensorElements = keras.layers.Dense( (inputSize**2 - inputSize)//2 + inputSize ,dtype='float32')(x)
        # b = tfp.bijectors.FillTriangular()
        tensor = FillTriangular()(tensorElements)        
        diagonal = MULTIPLY(inputSize)(tensor,)
        positiveTensor = keras.layers.Add()([tensor,2*ABS()(diagonal)])
        # tensor = keras.layers.Reshape((-1,inputSize,inputSize))(tensor)
        #transposed= keras.layers.EinsumDense("abcd->abdc",(-1,-1,inputSize,inputSize))(tensor)
        transposed= keras.layers.Lambda( (transpose) )(positiveTensor)
        
        symetric = positiveTensor*transposed
        self.nn = keras.Model(inputs = input,outputs = symetric)
        self.nn.compile("adam",loss="mse")
        self.pathfinder = pathfinder
        self.optimizer = tf.keras.optimizers.Adam()
    def __call__(self,x,*args: Any, **kwds: Any) -> Any:
        return self.nn(x)
    
    def load_model(self,path = "model" ):
        with open(path+".json", 'r') as plik:
            loaded_model_json = plik.read()
                
        model = keras.models.model_from_json(loaded_model_json)
        model.load_weights(path+".h5")
        self.nn = model
    def train(self,x,y,epochs=10,init_n=0):
        self.optimizer=keras.optimizers.Adam(learning_rate=0.0001)
        for f in range(epochs):
            
            geodesic =[]
            to_delete_x = []
            for p_index,pair in enumerate(x):
                # print(p_index)
                delta = (pair[1]-pair[0])/N
                if all(delta==0) and f ==0:
                    y=np.delete(y,p_index)
                    to_delete_x.append(p_index)
                    continue
                points = np.array([ pair[0] + delta*i for i in range(N+1) ])
                # self.pathfinder.__init__(points,self.nn,pop_size=1,scale=0.005,epochs=20+3*f)
                # # history = self.pathfinder.fit()
                # history = self.pathfinder.fit_all()
                # score,curve = zip(*history)
                
                # geodesic.append(curve[np.argmin(score)].T)
                geodesic.append(points.T)
            if f ==0 and init_n!=0:
                self.pathfinder.__init__(np.array(geodesic),None,pop_size=1,scale=0.005,epochs=160)
                history = self.pathfinder.fit_all()
            elif f<init_n:
                self.pathfinder.__init__(np.array(geodesic),None,pop_size=1,scale=0.005,epochs=30+1*f)
            elif f ==0:
                self.pathfinder.__init__(np.array(geodesic),self.nn,pop_size=1,scale=0.005,epochs=130)
                
                history = self.pathfinder.fit_all()
            else:
                
                
                history = self.pathfinder.fit_all(epochs=60)
                
            score,curve = zip(*history)
            geodesic = [np.array(c).T for c in curve[np.argmin(score)]]
            print(np.min(score))
            if len(to_delete_x) >0:
                for d in to_delete_x:
                    x = np.delete(x,d,axis=0)
            geodesic = tf.convert_to_tensor(geodesic,dtype=tf.float32)
            
            for e in range(10*5+5):
                with  tf.GradientTape() as tape:
                    
                    tape.watch(geodesic)
                    tensors = self.nn(geodesic,training=True)#
                    dx_i = []
                    tensors_i = (tensors[:,1:]+tensors[:,:-1])/2.
                    diff = geodesic[:,1:]-geodesic[:,:-1]
                    lenghts = (tf.einsum("ijk,ijkl,ijl->ij",diff,tensors_i,diff))
                    lenghts = tf.einsum("ij->i",lenghts)
                    # for i,path in enumerate(geodesic):
                    #     dx_j = []
                    #     diff = path[1:]-path[:-1]
                    #     tensor_i = (tensors[i][1:]+tensors[i][:-1])/2
                    #     dx_j2 = tf.einsum("ki,kij,kj->k",diff,tensor_i,diff)

                        
                    #     dx_i.append( tf.reduce_sum( tf.sqrt(dx_j2)) )
                        
                    # # mse = 
                    diif = tf.math.squared_difference(lenghts,y)
                    # diff_np = diif.numpy()
                    mse = tf.sqrt(tf.reduce_mean(diif))
                    gradients = tape.gradient(mse,self.nn.trainable_weights)
                    
                    self.optimizer.apply_gradients(zip(gradients, self.nn.trainable_weights))
                    print(mse)
            # self.nn.optimizer.appl
    def score(self,x,y,Npoints=30,epochs=150,training=False):
        
            
        geodesic =[]
        to_delete_x = []
        for p_index,pair in enumerate(x):
            # print(p_index)
            delta = (pair[1]-pair[0])/Npoints
            if all(delta==0):
                y=np.delete(y,p_index)
                to_delete_x.append(p_index)
                continue
            points = np.array([ pair[0] + delta*i for i in range(Npoints+1) ])
            geodesic.append(points.T)
        
        self.pathfinder.__init__(np.array(geodesic),self.nn,pop_size=1,scale=0.005,epochs=epochs)
        
        history = self.pathfinder.fit_all()
    
            
        score,curve = zip(*history)
        geodesic = [np.array(c).T for c in curve[np.argmin(score)]]
        print(np.min(score))
        if len(to_delete_x) >0:
            for d in to_delete_x:
                x = np.delete(x,d,axis=0)
        geodesic = tf.convert_to_tensor(geodesic,dtype=tf.float32)
        
        
        tensors = self.nn(geodesic,training=training)
        tensors_i = (tensors[:,1:]+tensors[:,:-1])/2.
        diff = geodesic[:,1:]-geodesic[:,:-1]
        lenghts = (tf.einsum("ijk,ijkl,ijl->ij",diff,tensors_i,diff))
        lenghts = tf.einsum("ij->i",lenghts)
        diif = tf.math.squared_difference(lenghts,y)
        mse = tf.sqrt(tf.reduce_mean(diif))
        print(mse.numpy())
        return mse.numpy()
    
    
    def measure(self,x,Npoints=30,epochs=150,training=False):
        
            
        geodesic =[]
        to_delete_x = []
        for p_index,pair in enumerate(x):
            # print(p_index)
            delta = (pair[1]-pair[0])/Npoints
            if all(delta==0):
               
                to_delete_x.append(p_index)
                continue
            points = np.array([ pair[0] + delta*i for i in range(Npoints+1) ])
            geodesic.append(points.T)
        
        self.pathfinder.__init__(np.array(geodesic),self.nn,pop_size=1,scale=0.005,epochs=epochs)
        
        history = self.pathfinder.fit_all()
    
            
        score,curve = zip(*history)
        geodesic = [np.array(c).T for c in curve[-1]]
        print(np.min(score))
        if len(to_delete_x) >0:
            for d in to_delete_x:
                x = np.delete(x,d,axis=0)
        geodesic = tf.convert_to_tensor(geodesic,dtype=tf.float32)
        
        
        tensors = self.nn(geodesic,training=training)
        tensors_i = (tensors[:,1:]+tensors[:,:-1])/2.
        diff = geodesic[:,1:]-geodesic[:,:-1]
        lenghts = (tf.einsum("ijk,ijkl,ijl->ij",diff,tensors_i,diff))
        lenghts = tf.einsum("ij->i",lenghts)

        return lenghts.numpy()
        
def save_model(name,model):
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+".h5")
    print("Saved model to disk")          
if __name__=="__main__":
    sp = GradientDescentCurve(np.array([0,1]),None,pop_size=2,epochs=50,scale=0.01)
    tens = tensorNetwork(2,[50,50,10],sp)
    # tens.load_model()
    
    sg = DataGenerator(1)
    points = sg.generate_2d_internal_sphere(50)
    pointsPair,dists = sg.distances(points,2)


    # tens.train(pointsPair,dists,10,10)
    tens.train(pointsPair,dists,15)
    save_model("model",tens.nn)
    
    
    points = sg.generate_2d_internal_sphere(55)
    pointsPair,dists = sg.distances(points,2)

    # 
    tens.train(pointsPair,dists)

    model_json = tens.nn.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    tens.nn.save_weights("model.h5")
    print("Saved model to disk")