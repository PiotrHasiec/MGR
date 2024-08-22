
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

        self.optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        self.nn.compile("adam",loss=self.geodesic_loss)
        self.pathfinder = pathfinder
        
    def __call__(self,x,*args: Any, **kwds: Any) -> Any:
        return self.nn(x)
    
    def load_model(self,path = "model" ):
        with open(path+".json", 'r') as plik:
            loaded_model_json = plik.read()
                
        model = keras.models.model_from_json(loaded_model_json)
        model.load_weights(path+".weights.h5")
        self.nn = model
    
    def geodesic_loss(self,data,y_pred):
        y_true = data

        diff = self.geodesic[:,1:]-self.geodesic[:,:-1]
        tensor_avg= y_pred[:,1:]-y_pred[:,:-1]
        lenghts = (keras.ops.einsum("ijk,ijkl,ijl->ij",diff,tensor_avg,diff))
        lenghts = keras.ops.einsum("ij->i",lenghts)
        return keras.losses.mean_squared_error(y_true,lenghts)

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
                self.pathfinder.__init__(np.array(geodesic),None,pop_size=1,scale=0.005,epochs=10)
                history = self.pathfinder.fit_all()
            elif f<init_n:
                self.pathfinder.__init__(np.array(geodesic),None,pop_size=1,scale=0.005,epochs=30+1*f)
            elif f ==0:
                self.pathfinder.__init__(np.array(geodesic),self.nn,pop_size=1,scale=0.005,epochs=130)
                
                history = self.pathfinder.fit_all()
            else:
                
                
                history = self.pathfinder.fit_all(epochs=60)
                
            score,curve = zip(*history)
            # data = [(np.array(c).T ,y[i])for i,c in enumerate(curve[-1])]
            
            geodesic = [np.array(c).T for c in curve[-1]]
            print(np.min(score))
            if len(to_delete_x) >0:
                for d in to_delete_x:
                    x = np.delete(x,d,axis=0)
            self.geodesic = tf.convert_to_tensor(geodesic,dtype=tf.float32)
            # data = np.append(geodesic,y,1)
            self.nn.fit(self.geodesic,y,epochs=50,batch_size=len(geodesic))
            # self.nn.optimizer.appl
    def score(self,x,y,Npoints=30,epochs=150,training=False):
        
        self.nn.compile("adam",loss=self.geodesic_loss)
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
    model.save_weights(name+".weights.h5")
    print("Saved model to disk")          
if __name__=="__main__":
    sp = GradientDescentCurve(np.array([0,1]),None,pop_size=2,epochs=50,scale=0.01)
    tens = tensorNetwork(2,[250,250,30],sp)
    # tens.load_model()
    
    sg = DataGenerator(1)
    points = sg.generate_2d_internal_sphere(35)
    pointsPair,dists = sg.distances(points,3)

    save_model("model_gpu",tens.nn)
    tens.train(pointsPair,dists,5,5)
    tens.train(pointsPair,dists,15)
    save_model("model_gpu",tens.nn)
    
    
    points = sg.generate_2d_internal_sphere(55)
    pointsPair,dists = sg.distances(points,2)

    # 
    tens.train(pointsPair,dists)

    save_model("model_gpu",tens.nn)