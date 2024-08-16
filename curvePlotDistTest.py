import numpy as np
import tensorflow as tf
from DataGeneration import DataGenerator
from tensorNetwork import tensorNetwork
from EvolutionCurve import GradientDescentCurve
from matplotlib import pyplot as plt
if __name__ == "__main__":
    N =25
    from DataGeneration import DataGenerator
    sg = DataGenerator()
    points = sg.generate_2d_internal_sphere(60)
    pointsPair,dists = sg.distances(points,6)

    
    
    sp = GradientDescentCurve(np.array([0,1]),None,pop_size=2,epochs=50,scale=0.01)
    tens = tensorNetwork(2,[10,10,10],sp)
    def f(x):
        if x ==0 or x==N:
            return 0
        else:
            return 1
    tens.load_model("model")
    scores = []
    
    # score = tens.score(pointsPair,dists,training=True)
    # print("score: ", score)
    # path_init = np.array([ np.array([np.array(p[0])+j*(p[1]-p[0])/N for j in range(N+1)]).T for p in pointsPair])
    # # # p1,p2 = p[0],p[1]
    # # # p1,p2 = np.array([np.pi,0]),np.array([0,0])
    # # # path_init = np.array([p1+i*(p2-p1)/N for i in range(N+1)]).T
    # sp.__init__(path_init,tens.nn,1,250,0.0005)
    s =tens.measure(pointsPair)
    history = sp.fit_all()
    
    from matplotlib import pyplot as plt
    for s,p in history[-1:]:
        print(s)
        for path in p[np.random.randint(0,len(p),10)]:
            plt.plot(path[0],path[1],alpha = 0.5)
            plt.scatter(path[0],path[1])
            plt.scatter(path[0][-1],path[1][-1],c='r')
            plt.scatter(path[0][0],path[1][0],c='r')
        
    # s,p = history[-1]
    # l = sp.score(p[0])
    # for p in points:
    #     plt.scatter(p[0],p[1])
    

    # plt.plot(p[0][0],p[0][1],c = "r",alpha = 1.)
    plt.show()
    # p=np.swapaxes(p,2,1)
    
    # l = sp.score(p)
    for i in range(len(dists)):
        # l=s[i]
        # l = sp.score(s[i][None])
        print(dists[i],s[i],dists[i] -s[i])
        scores.append(dists[i] - s[i])

    print(np.sqrt(np.mean(np.square(scores))))
        