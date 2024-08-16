import numpy as np
import tensorflow as tf
from DataGeneration import DataGenerator
# from tensorNetwork import tensorNetwork
from scipy.interpolate import BSpline,splprep,splev
from tqdm import tqdm
from geneticalgorithm import geneticalgorithm as ga

import keras
class GradientDescentCurve:
    def __init__(self,knots,model,pop_size = 10, epochs = 50,scale = 0.1):
        self.knots = knots    
        self.scores = []
        self.model = model
        self.epochs = epochs
        self.population_size = pop_size
        self.scale = scale*10
        self.population = []
        for i in range(self.population_size):
            # self.population.append( splprep( knots,k=degree)[0])
            self.population.append( (0, knots.copy(), 0))
            self.mutate(i,True)
        self.scale = scale
        self.optimizer = keras.optimizers.Adam()
    def fit(self):
        history = []
        for i in range(self.population_size):
            optimizer = keras.optimizers.Adam(learning_rate=0.01)
            geodesic = [tf.Variable(tf.convert_to_tensor(var,dtype="float32")) for var in self.population[i][1].T ]
            
            for e in tqdm(range(self.epochs)):
                with  tf.GradientTape() as tape:        
                    tape.watch(geodesic)
                    # geodesic_mod=tf.math.floormod(geodesic,tf.ones_like(geodesic)*np.pi)
                    if self.model is None:
                        tensors = [np.eye(len(self.population[i][1]))]*len(self.population[i][1].T)
            
                    else:
                        path = self.population[i][1].T[np.newaxis]
                        tensors =self.model(path)
                        tensors = tf.squeeze(tensors)
                    dx_j = []
                    
                    
                    diff = tf.convert_to_tensor([ai-bi for ai,bi in zip(geodesic[1:],geodesic[:-1])])
                    tensor_i = (tensors[1:]+tensors[:-1])/2.
                    dx_j2 = tf.einsum("ki,kij,kj->k",diff,tensor_i,diff)
                    
                    # for j in range( len(geodesic)-1):
                    #     v = tf.reshape(geodesic[j]-geodesic[j+1],(-1,1))
                    #     # vT = tf.transpose(v)
                    #     tensor_ij = ((tensors[j]+tensors[j+1])/2)
                    #     # vTM = tf.matmul(vT,tensor_ij)
                    #     # vTMv = tf.matmul(vTM,v)
                    #     vTMv2 = tf.einsum("ji,jk,kl->il",v,tensor_ij,v)
                        
                    #     dx_j.append(vTMv2)
                    lenght = ( tf.reduce_sum( tf.sqrt(dx_j2)) )
                    # print(lenght.numpy())
                    
                    gradients = tape.gradient(lenght,geodesic[1:-1])
                    
                    optimizer.apply_gradients(zip(gradients, geodesic[1:-1]))
                self.population[i] = (0,tf.convert_to_tensor(tf.math.floormod(geodesic,tf.ones_like(geodesic)*np.pi*2)).numpy().T,0)
            history.append((lenght.numpy(),tf.convert_to_tensor(geodesic).numpy().T))#
        return history
    
    def fit_all(self,epochs = None):
            history = []
            population = self.knots
            optimizer = keras.optimizers.AdamW(learning_rate=0.01)
            
            geodesic = [tf.Variable(population.T[0],dtype="float32",trainable=False) ]+[tf.Variable(var,dtype="float32") for var in population.T[1:-1] ]+[tf.Variable(population.T[-1],dtype="float32",trainable=False) ]
            if epochs == None:
                epochs=self.epochs
            for e in tqdm(range(epochs)):
                with  tf.GradientTape() as tape:        
                    tape.watch(geodesic)
                    geodesic2 = tf.transpose(tf.stack(geodesic,axis=1))
                    # geodesic_mod=tf.math.floormod(geodesic,tf.ones_like(geodesic)*np.pi)
                    if self.model is None:
                        tensors = np.array([[np.eye(len(self.knots[0]))]*len(self.knots[0].T)]*len(self.knots))
            
                    else:
                        # path = self.population[i][1].T[np.newaxis]
                        tensors =self.model(geodesic2)
                        tensors = tensors
            #         dx_j = []
                    # diff = tf.convert_to_tensor([ai-bi for ai,bi in zip(geodesic2[:,1:],geodesic2[:,:-1])])
                    diff = geodesic2[:,1:]-geodesic2[:,:-1]
                    tensors_i = (tensors[:,1:]+tensors[:,:-1])/2.
                    lenghts = (tf.einsum("ijk,ijkl,ijl->ij",diff,tensors_i,diff))
                    lenghts = tf.einsum("ij->i",lenghts)
            #         diff = tf.convert_to_tensor([ai-bi for ai,bi in zip(geodesic[1:],geodesic[:-1])])
            #         tensor_i = (tensors[1:]+tensors[:-1])/2.
            #         dx_j2 = tf.einsum("ki,kij,kj->k",diff,tensor_i,diff)
                    
            #         # for j in range( len(geodesic)-1):
            #         #     v = tf.reshape(geodesic[j]-geodesic[j+1],(-1,1))
            #         #     # vT = tf.transpose(v)
            #         #     tensor_ij = ((tensors[j]+tensors[j+1])/2)
            #         #     # vTM = tf.matmul(vT,tensor_ij)
            #         #     # vTMv = tf.matmul(vTM,v)
            #         #     vTMv2 = tf.einsum("ji,jk,kl->il",v,tensor_ij,v)
                        
            #         #     dx_j.append(vTMv2)
            #         lenght = ( tf.reduce_sum( tf.sqrt(dx_j2)) )
            #         # print(lenght.numpy())
                    
                    gradients = tape.gradient(lenghts,geodesic[1:-1])
                    
                    # min_ = tf.argmin(lenghts).numpy()
                    optimizer.apply_gradients(zip(gradients,geodesic[1:-1]))
                    history.append(( lenghts.numpy(),np.array(geodesic).T ))
            #     self.population[i] = (0,tf.convert_to_tensor(tf.math.floormod(geodesic,tf.ones_like(geodesic)*np.pi*2)).numpy().T,0)
            # history.append((lenght.numpy(),tf.convert_to_tensor(geodesic).numpy().T))#
            # return history
            # print(lenghts[min_].numpy())
            self.knots = history[-1][1]
            return history
    def mutate(self,i, iffirst = False):
        spline = self.population[i][1]
        # for coof_i in range( len(spline)):
        #     if np.random.uniform(0,1) <= 0.05:
        #         self.population[i][1][coof_i][1:-1] += np.random.normal(0,self.scale,len(self.population[i][1][coof_i][1:-1]))
        ch = self.population[i][1].T
        for j in range(1,len(spline.T)-2):
            if np.random.uniform(0,1) <= 0.05 :
                ch[j],ch[j+1]    = ch[j],ch[j+1]
            if np.random.uniform(0,1) <= 0.05 or iffirst:
                ch[j] += np.random.normal(0,self.scale, np.shape(ch[j]) )
        return spline
    def score(self,i,n_points=11):
        
        obj = self.population[i]
        # spline = splev( np.linspace(0,1,n_points),obj)
        spline = self.population[i][1]
        spline = np.array(spline)
        if self.model is None:
            tensors = [np.eye(len(spline))]*len(spline.T)
            
        else:
            toModel = spline.T[np.newaxis]
            tensors = self.model(toModel).numpy()[0]
        spline = spline.T
        # dx_i = [np.matmul(np.matmul( np.array(spline[i]-spline[i+1]).T,np.array(tensors[i]) ),np.array(spline[i]-spline[i+1]) ) for i in range(len(spline)-1)]
        dx_i = []
        for i in range(len(spline)-1):
            v = np.array(spline[i]-spline[i+1])
            vTM = np.matmul( v.T,np.array(tensors[i]))
            vTMv = np.matmul(vTM,v)
            dx_i.append(vTMv)
            
        
        lenght = np.sum((dx_i))
        
        return lenght
    
    def score(self,path,n_points=11):
        
        spline = path
        spline = np.array(spline)
        if self.model is None:
            tensors = [np.eye(len(spline))]*len(spline.T)
            
        else:
            toModel = spline#[np.newaxis]
            tensors = self.model(toModel)
            tensors = (tensors[:,1:]+tensors[:,:-1])/2.
        spline = spline.T
        # dx_i = [np.matmul(np.matmul( np.array(spline[i]-spline[i+1]).T,np.array(tensors[i]) ),np.array(spline[i]-spline[i+1]) ) for i in range(len(spline)-1)]
        dx_i = []
        
        diff = toModel[:,1:]-toModel[:,:-1]
    
        lenghts = tf.einsum("ijk,ijkl,ijl->i",diff,tensors,diff)
        # for i in range(len(spline)-1):
        #     v = np.array(spline[i]-spline[i+1])
        #     vTM = np.matmul( v.T,np.array(tensors[i]))
        #     vTMv = np.matmul(vTM,v)
        #     dx_i.append(vTMv)
            
        
        # lenght = np.sum(lenghts.numpy())
        
        return lenghts.numpy()
        
class EvolutionaryCurve_:
    def __init__(self,knots,model,pop_size = 400, epochs = 10,scale = 0.1,disp =1) -> None:
        varbonds = []
        self.start = np.array(knots).T[0]
        self.end = np.array(knots).T[-1]
        algorithm_param = {'max_num_iteration': 300,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.001,\
                   'crossover_probability': 0.05,\
                   'parents_portion': 0.01,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}
        self.model = model
        self.dim = len(knots)
        for i in range(len(knots)):
            varbonds += [ [k-disp,k+disp] for k in knots[i][0:-2] ] 
        vartype = 'real'
        varbonds=np.array(varbonds)
        self.ga = ga(self.score,dimension= len(varbonds),variable_type=vartype,variable_boundaries=varbonds,algorithm_parameters=algorithm_param)
        
    def score(self,X):
            

            spline = np.reshape(X, (-1,self.dim))
            spline=np.insert(spline,0,self.start,axis=0)
            spline=np.append(spline, np.reshape(self.end, (1,-1)),axis=0)
            spline = np.array(spline).T
           
            if self.model is None:
                tensors = [np.eye(len(spline))]*len(spline.T)
                
            else:
                toModel = spline.T[np.newaxis]
                tensors = self.model(toModel).numpy()[0]
            spline = spline.T
            # dx_i = [np.matmul(np.matmul( np.array(spline[i]-spline[i+1]).T,np.array(tensors[i]) ),np.array(spline[i]-spline[i+1]) ) for i in range(len(spline)-1)]
            dx_i = []
            for i in range(len(spline)-1):
                v = np.array(spline[i]-spline[i+1])
                vTM = np.matmul( v.T,np.array(tensors[i]))
                vTMv = np.matmul(vTM,v)
                dx_i.append(vTMv)
                
            
            lenght = np.sum(np.sqrt(dx_i))
            
            return lenght
class EvolutionaryCurve:
    def __init__(self,knots,model,pop_size = 400, epochs = 10,scale = 0.1) -> None:
        self.knots = knots    
        self.scores = []
        self.model = model
        self.epochs = epochs
        self.population_size = pop_size
        self.scale = scale*5
        self.population = []
        for i in range(self.population_size):
            # self.population.append( splprep( knots,k=degree)[0])
            self.population.append( (0, knots.copy(), 0))
            self.mutate(i,True)
        self.scale = scale
            
    def fit(self,pop_size = 10, epochs = 10):
        history = []
        for e in tqdm(range(self.epochs)):
            scores = []
            for i in range(self.population_size):
                    scores.append( (i,self.score(i)) )
            scores.sort(key=lambda a: a[1])
            # print(scores[:1])
            
            new_population = []
            
            [ind,scoresval] = zip(*scores)
            history.append( (scoresval[0],self.population[scores[0][0]][1].copy())  )
            
            p = scoresval/np.sum(scoresval)
            ranking = np.cumsum(p)
            
            if e != epochs-1:
                for i in range(self.population_size):
                    partner1 = np.searchsorted(ranking,np.random.uniform(0,1))
                    partner2 = np.searchsorted(ranking,np.random.uniform(0,1))
                    
                    new_population.append(self.cross(ind[partner1],ind[partner2]))
                self.population = new_population.copy()
           
                
        self.scores=scores
        print(np.min(scoresval))
        return history
            
    
    def mutate(self):
        for spline_i,spline in enumerate(self.population):
            for node_i in range( len(spline)):
                self.population[spline_i][0][node_i] += np.random.normal(0,self.scale)
                
    def mutate(self,i, iffirst = False):
        spline = self.population[i][1]
        # for coof_i in range( len(spline)):
        #     if np.random.uniform(0,1) <= 0.05:
        #         self.population[i][1][coof_i][1:-1] += np.random.normal(0,self.scale,len(self.population[i][1][coof_i][1:-1]))
        ch = self.population[i][1].T
        for j in range(1,len(spline.T)-2):
            if np.random.uniform(0,1) <= 0.05 :
                ch[j],ch[j+1]    = ch[j],ch[j+1]
            if np.random.uniform(0,1) <= 0.05 or iffirst:
                ch[j] += np.random.normal(0,self.scale, np.shape(ch[j]) )
        return spline
                
    def cross(self,i,j, inplace = True):
        if inplace == True:
            for k in range(len(self.population[i][1])): self.population[i][1][k] = (self.population[i][1][k]+ self.population[j][1][k])/2
            return (0,self.population[i][1].copy(),0)
        else:
            child = (0,self.population[i][1].copy(),0)
            for k in range(len(self.population[i][1])): child[1][k] = (self.population[i][1][k]+ self.population[j][1][k])/2
            return child
    def score(self,i,n_points=11):
        
        obj = self.population[i]
        # spline = splev( np.linspace(0,1,n_points),obj)
        spline = self.population[i][1]
        spline = np.array(spline)
        if self.model is None:
            tensors = [np.eye(len(spline))]*len(spline.T)
            
        else:
            toModel = spline.T[np.newaxis]
            tensors = self.model(toModel).numpy()[0]
        spline = spline.T
        # dx_i = [np.matmul(np.matmul( np.array(spline[i]-spline[i+1]).T,np.array(tensors[i]) ),np.array(spline[i]-spline[i+1]) ) for i in range(len(spline)-1)]
        dx_i = []
        for i in range(len(spline)-1):
            v = np.array(spline[i]-spline[i+1])
            vTM = np.matmul( v.T,np.array(tensors[i]))
            vTMv = np.matmul(vTM,v)
            dx_i.append(vTMv)
            
        
        lenght = np.sum(np.sqrt(dx_i))
        
        return lenght
    
if __name__ == "__main__":
    N = 10
    
    def f(x):
        if x ==0 or x==N:
            return 0
        else:
            return 1
    sg = DataGenerator()
    points = sg.generate_2d_internal_sphere(20)
    pointsPair,dists = sg.distances(points,4)
    points = np.array([ np.array([np.array(p[0])+np.random.uniform(-0.2,0.2,2)*f(j)+j*(p[1]-p[0])/N for j in range(N+1)]).T for p in pointsPair])
    # points = np.array( [np.array([ [0+np.random.uniform(-0.4,0.4)*f(i),((2*np.pi+0.1)*(float(i)/N ))+np.random.uniform(-0.4,0.4)*f(i) ] for i in range(N+1) ]).T for i in range(10000)] )
    # # sp = GradientDescentCurve(points, None,pop_size=10,epochs=40,scale=0.001)
    # # o=sp.fit()
    
    
    sp = GradientDescentCurve(points,None,pop_size=1,epochs=1000,scale=0.01)
    # tens = tensorNetwork(2,[10,10,10],sp)
    # # spline=np.insert(spline,0,self.start,axis=0)
    # # spline=np.append(spline, np.reshape(self.end, (1,-1)),axis=0)
    # tens.load_model()
    # for i,p in enumerate(pointsPair): 
    #     print(dists[i])

    # print(sp.score(1))
    # sp.mutate(1)
    # print(sp.score(1))
    # sp.cross(1,0)

    # print(sp.score(1))
    history = sp.fit_all()
    
    from matplotlib import pyplot as plt
    for s,p in history[-1:]:
        print(s)
        for path in p[np.random.randint(0,len(p),10)]:
            plt.plot(path[0],path[1],alpha = 0.5)
            plt.scatter(path[0],path[1])
            plt.scatter(path[0][-1],path[1][-1],c='r')
            plt.scatter(path[0][0],path[1][0],c='r')
        
    s,p = history[-1]
    # for p in points:
    #     plt.scatter(p[0],p[1])


    # plt.plot(p[0][0],p[0][1],c = "r",alpha = 1.)
    plt.show()
    # aa = sp.population[0]
    # print(sp.population[sp.scores[0][0]][0])
    # print(sp.population[sp.scores[0][0]][1])
    # print(sp.population[sp.scores[0][0]][2])
