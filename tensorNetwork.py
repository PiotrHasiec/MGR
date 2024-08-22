
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
N = 40

@keras.saving.register_keras_serializable()
def transpose(w):
    return  tf.einsum("abcd->abdc",w)

@keras.saving.register_keras_serializable()
class FillTriangular(keras.Layer):
    def call(self, x):
        return tfp.math.fill_triangular(x)
@keras.saving.register_keras_serializable()
class ABS(keras.Layer):
    def call(self, x):
        return tf.abs(x)

@keras.saving.register_keras_serializable()
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
        # with open(path+".keras", 'r') as plik:
        #     loaded_model_json = plik.read()
                
        model = keras.models.load_model(path+".keras")
        model.load_weights(path+".weights.h5")
        self.nn = model
    def train(self,x,y,epochs=4,epochs2=5,init_n=0,batch_size=32,batch_parts = 1):
        self.optimizer=keras.optimizers.Adadelta(learning_rate=1.)
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
                self.pathfinder.__init__(np.array(geodesic),None,pop_size=1,scale=0.005,epochs=100)
                history = self.pathfinder.fit_all(batch_parts=batch_parts)
            elif f<init_n:
                self.pathfinder.__init__(np.array(geodesic),None,pop_size=1,scale=0.005,epochs=30+1*f)
            elif f ==0:
                self.pathfinder.__init__(np.array(geodesic),self.nn,pop_size=1,scale=0.005,epochs=500)
                
                history = self.pathfinder.fit_all(batch_parts=batch_parts)
            else:
                
                
                history = self.pathfinder.fit_all(epochs=60,batch_parts=batch_parts)
                
            score,curve = zip(*history)
            geodesic = [np.array(c).T for c in curve]
            # print(np.min(score))
            if len(to_delete_x) >0:
                for d in to_delete_x:
                    x = np.delete(x,d,axis=0)
            geodesic = tf.convert_to_tensor(geodesic,dtype=tf.float32)
            
            for e in range(epochs2):
                mmse = []
                for bindex in tqdm(range(len(geodesic)//batch_size)):

                    with  tf.GradientTape() as tape:
                        batch = geodesic[batch_size*bindex:batch_size*(bindex+1)]
                        tape.watch(batch)
                        tensors = self.nn(batch,training=True)#
                        dx_i = []
                        tensors_i = (tensors[:,1:]+tensors[:,:-1])/2.
                        diff = batch[:,1:]-batch[:,:-1]
                        lenghts = tf.sqrt(tf.einsum("ijk,ijkl,ijl->ij",diff,tensors_i,diff))
                        lenghts = (tf.einsum("ij->i",lenghts))
                        # for i,path in enumerate(geodesic):
                        #     dx_j = []
                        #     diff = path[1:]-path[:-1]
                        #     tensor_i = (tensors[i][1:]+tensors[i][:-1])/2
                        #     dx_j2 = tf.einsum("ki,kij,kj->k",diff,tensor_i,diff)

                            
                        #     dx_i.append( tf.reduce_sum( tf.sqrt(dx_j2)) )
                        y_batch =y[batch_size*bindex:batch_size*(bindex+1)]
                        # # mse = 
                        diif = tf.math.squared_difference(lenghts,y_batch)
                        # diff_np = diif.numpy()
                        mse = tf.reduce_mean(diif)
                        gradients = tape.gradient(mse,self.nn.trainable_weights)
                        
                        self.optimizer.apply_gradients(zip(gradients, self.nn.trainable_weights))
                        mmse.extend(diif)
                print(np.sqrt(np.mean(mmse)))
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
        geodesic = [np.array(c).T for c in curve[-1]]
        print(np.min(score))
        if len(to_delete_x) >0:
            for d in to_delete_x:
                x = np.delete(x,d,axis=0)
        geodesic = tf.convert_to_tensor(geodesic,dtype=tf.float32)
        
        
        tensors = self.nn(geodesic,training=training)
        tensors_i = (tensors[:,1:]+tensors[:,:-1])/2.
        diff = geodesic[:,1:]-geodesic[:,:-1]
        lenghts = tf.sqrt(tf.einsum("ijk,ijkl,ijl->ij",diff,tensors_i,diff))
        lenghts = tf.einsum("ij->i",lenghts)
        diif = tf.math.squared_difference(lenghts,y)
        mse = tf.sqrt(tf.reduce_mean(diif))
        print(mse.numpy())
        return mse.numpy()
    
    
    def measure(self,x,Npoints=30,epochs=190,training=False,batch_size = 256,return_paths = False,curve_fit_batch_parts = 1):
        
            
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
            # self.pathfinder.__init__(points,self.nn,pop_size=1,scale=0.005,epochs=20+3*f)
            # # history = self.pathfinder.fit()
            # history = self.pathfinder.fit_all()
            # score,curve = zip(*history)
            
            # geodesic.append(curve[np.argmin(score)].T)
            geodesic.append(points.T)

        self.pathfinder.__init__(np.array(geodesic),self.nn,pop_size=1,scale=0.005,epochs=200)

            
        history = self.pathfinder.fit_all(epochs=epochs,batch_parts = curve_fit_batch_parts)
            
        score,curve = zip(*history)
        geodesic = [np.array(c).T for c in curve]
        # print(np.min(score))
        if len(to_delete_x) >0:
            for d in to_delete_x:
                x = np.delete(x,d,axis=0)
        geodesic = tf.convert_to_tensor(geodesic,dtype=tf.float32)
        
        
        # tensors = self.nn(geodesic,training=training)
        # tensors_i = (tensors[:,1:]+tensors[:,:-1])/2.
        # diff = geodesic[:,1:]-geodesic[:,:-1]
        # lenghts = (tf.einsum("ijk,ijkl,ijl->ij",diff,tensors_i,diff))
        # lenghts = tf.einsum("ij->i",lenghts)



        lenghts_flat = []
        for bindex in tqdm(range(len(geodesic)//batch_size + 1)):

            # with  tf.GradientTape() as tape:
                batch = geodesic[batch_size*bindex:batch_size*(bindex+1)]
                # tape.watch(batch)
                tensors = self.nn(batch,training=training)#
                dx_i = []
                tensors_i = (tensors[:,1:]+tensors[:,:-1])/2.
                diff = batch[:,1:]-batch[:,:-1]
                lenghts = tf.sqrt(tf.einsum("ijk,ijkl,ijl->ij",diff,tensors_i,diff))
                lenghts = (tf.einsum("ij->i",lenghts))
                lenghts_flat.extend(lenghts)
                # for i,path in enumerate(geodesic):
                #     dx_j = []
                #     diff = path[1:]-path[:-1]
                #     tensor_i = (tensors[i][1:]+tensors[i][:-1])/2
                #     dx_j2 = tf.einsum("ki,kij,kj->k",diff,tensor_i,diff)

                    
                #     dx_i.append( tf.reduce_sum( tf.sqrt(dx_j2)) )
                # y_batch =y[batch_size*bindex:batch_size*(bindex+1)]
                # # mse = 
                # diif = tf.math.squared_difference(lenghts,y_batch)
                # diff_np = diif.numpy()
                # mse = tf.reduce_mean(diif)
                # gradients = tape.gradient(mse,self.nn.trainable_weights)
                
                # self.optimizer.apply_gradients(zip(gradients, self.nn.trainable_weights))
                # mmse.append(mse)
        # print(np.sqrt(np.mean(mmse)))
        if not return_paths:
            return np.array(lenghts_flat)
        return (np.array(lenghts_flat),geodesic.numpy())
        
def save_model(name,model):
    # model_json = model.to_json()
    # with open(name+".keras", "w") as json_file:
    #     json_file.write(model_json)
    # serialize weights to HDF5
    model.save(name+".keras")
    model.save_weights(name+".weights.h5")
    print("Saved model to disk")          
if __name__=="__main__":
    sp = GradientDescentCurve(np.array([0,1]),None,pop_size=2,epochs=70,scale=0.01)
    tens = tensorNetwork(2,[200,100,50],sp)
    
    
    sg = DataGenerator(1)
    points = sg.generate_2d_internal_sphere(70)
    pointsPair,dists = sg.distances(points,2)

    # save_model("model_gpu_sqrt",tens.nn)
    tens.load_model("model_gpu_sqrt")
    # tens.train(pointsPair,dists,epochs=3,init_n=3)
    tens.train(pointsPair,dists,4,batch_size=16,batch_parts=3)
    save_model("model_gpu_sqrt",tens.nn)
    
    
    points = sg.generate_2d_internal_sphere(80)
    pointsPair,dists = sg.distances(points,2)

    # 
    tens.train(pointsPair,dists,batch_parts=4)

    save_model("model_gpu_sqrt",tens.nn)