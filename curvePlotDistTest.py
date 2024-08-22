import numpy as np
import tensorflow as tf
from DataGeneration import DataGenerator
from tensorNetwork import tensorNetwork
from EvolutionCurve import GradientDescentCurve
from matplotlib import pyplot as plt
from keras.src.saving import serialization_lib
import keras
serialization_lib.enable_unsafe_deserialization()
def grid_gen(x,y,N1,N2=None):
    x_min,x_max = x[0],x[1]
    y_min,y_max = y[0],y[1]
    if N2 == None:
         N2=N1//2
    x_list = [x_min + i*(x_max-x_min)/N1 for i in range(N1+1)]
    y_list = [y_min + i*(y_max-y_min)/N2 for i in range(N2+1)]

    pointsPair  = []
    # for i in range(N1+1):
    #      pointsPair.append( [[x_list[i],y_list[0]],[x_list[i],y_list[-1]]])
    # for j in range(N2+1): 
    #     pointsPair.append( [[x_list[0],y_list[j]],[x_list[-1],y_list[j]]])
    for i in range(N1+1):
          for j in range(N2+1):
                if j>0:
                    pointsPair.append([[x_list[i],y_list[j]],[x_list[i],y_list[j-1]]])
                if i>0:
                    pointsPair.append([[x_list[i],y_list[j]],[x_list[i-1],y_list[j]]])
                if i<N1:
                     pointsPair.append([[x_list[i],y_list[j]],[x_list[i+1],y_list[j]]])
                if j<N2:
                    pointsPair.append([[x_list[i],y_list[j]],[x_list[i],y_list[j+1]]])
                
    return np.array(pointsPair)

def prepare_paths(x,Npoints):
        geodesic = []
        for p_index,pair in enumerate(x):
            delta = (pair[1]-pair[0])/Npoints
            points = np.array([ pair[0] + delta*i for i in range(Npoints+1) ])
            geodesic.append(points.T)
        return geodesic

def tensor_on_sphere(x):
    to_return = []
    for path in x:
        tensor_on_path =[]
        for point in path:
              tensor_on_path.append(np.array([[1,0],[0,(np.sin(point[0]))**2]])*0.25) 
        to_return.append(np.array(tensor_on_path))
    return np.array(to_return)
if __name__ == "__main__":
    N =25
    from DataGeneration import DataGenerator
    sg = DataGenerator()
    points = sg.generate_2d_internal_sphere(3)
    pointsPair,dists = sg.distances(points,1)

    
    
    sp = GradientDescentCurve(np.array([0,1]),None,pop_size=1,epochs=30,scale=0.01)
    tens = tensorNetwork(2,[10,10,10],sp)
    def f(x):
        if x ==0 or x==N:
            return 0
        else:
            return 1
    tens.load_model("model_gpu_sqrt")
    # tens.train(pointsPair,dists)
    
    
    # score = tens.score(pointsPair,dists,training=True)
    # print("score: ", score)
    # path_init = np.array([ np.array([np.array(p[0])+j*(p[1]-p[0])/N for j in range(N+1)]).T for p in pointsPair])
    # # # p1,p2 = p[0],p[1]
    # p1,p2 = np.array([np.pi,0]),np.array([0,0])
    # path_init = np.array([p1+i*(p2-p1)/N for i in range(N+1)]).T
    # sp.__init__(path_init,tens.nn,1,250,0.0005)
    pointsPair = grid_gen((0,np.pi),(0.,2*np.pi),20,20)
    pointsPair2 = grid_gen((0.1,np.pi),(np.pi,2*np.pi),5,5)
    # pointsPair3 = np.concatenate([pointsPair,pointsPair2],axis=0)
    
    paths_init = prepare_paths(pointsPair,5)

    
    tensors_t = tensor_on_sphere(np.swapaxes(np.array(paths_init),-1,-2))
    tensors_p = tens.nn(np.swapaxes(np.array(paths_init),-1,-2))

    print(tensors_t[:3])
    print(tensors_p[:3])

    print(keras.metrics.mean_squared_error( np.reshape(tensors_t,(-1,)),np.reshape(tensors_p,(-1,))))
    sp.__init__(np.array(paths_init),tensor_on_sphere,1,250,0.0005)
    # sc,paths=zip(*sp.fit_all(30)) 
    sc2,paths2 =tens.measure(pointsPair,epochs=300,training=False,return_paths=True,Npoints=5,curve_fit_batch_parts = 1)
    # scores = tens.score(pointsPair,dists)
    # history = sp.fit_all()
    
    from matplotlib import pyplot as plt

    # for p in paths:
    #         # p = p.T
       
    #         plt.plot(p[0],p[1],alpha = 0.5,c='b')
    #         # plt.scatter(p[0],p[1])
    #         plt.scatter(p[0][-1],p[1][-1],c='r')
    #         plt.scatter(p[0][0],p[1][0],c='r') 
    for p in paths2:
            p = p.T
       
            plt.plot(p[0],p[1],alpha = 0.5,c='g')
            # plt.scatter(p[0],p[1])
            plt.scatter(p[0][-1],p[1][-1],c='r')
            plt.scatter(p[0][0],p[1][0],c='r')
    plt.show()

    for p in paths:
            # p = p.T
       
            plt.plot(p[0],p[1],alpha = 0.5,c='b')
            # plt.scatter(p[0],p[1])
            plt.scatter(p[0][-1],p[1][-1],c='r')
            plt.scatter(p[0][0],p[1][0],c='r')
    plt.show()  
    for p in paths2:
            p = p.T
       
            plt.plot(p[0],p[1],alpha = 0.5,c='g')
            # plt.scatter(p[0],p[1])
            plt.scatter(p[0][-1],p[1][-1],c='r')
            plt.scatter(p[0][0],p[1][0],c='r')
    plt.show()
    # s,p = history[-1]
    # l = sp.score(p[0])
    # for p in points:
    #     plt.scatter(p[0],p[1])
    

    # plt.plot(p[0][0],p[0][1],c = "r",alpha = 1.)

    # p=np.swapaxes(p,2,1)
    scores = []
    # l = sp.score(p)
    # for i in range(len(dists)):
        # l=s[i]
        # l = sp.score(s[i][None])
        # print(dists[i],s[i],dists[i] -s[i])
        # scores.append(dists[i] - sc[i])
        # scores.append()

    
    print("mape: ",keras.metrics.mean_absolute_percentage_error(dists,sc))
    print("mse: ",keras.metrics.mean_squared_error(dists,sc)) 
    print("mean squared dist: ",np.sqrt(np.mean(dists**2))) 

    print("mse/msd: ",keras.metrics.mean_squared_error(dists,sc)/np.sqrt(np.mean(dists**2))) 


    print("mape: ",keras.metrics.mean_absolute_percentage_error(dists,sc2))
    print("mse: ",keras.metrics.mean_squared_error(dists,sc2)) 
    print("mean squared dist: ",np.sqrt(np.mean(dists**2))) 

    print("mse/msd: ",keras.metrics.mean_squared_error(dists,sc2)/np.sqrt(np.mean(dists**2))) 