import numpy as np
import tensorflow as tf
from scipy.interpolate import BSpline,splprep,splev

class EvolutionaryCurve:
    def __init__(self,knots,degree,n_coof,model,pop_size = 400, epochs = 10,scale = 0.1) -> None:
        self.knots = knots
        self.degree = degree
        self.n_coof = n_coof
        self.scores = []
        self.model = model
        self.epochs = epochs
        self.population_size = pop_size
        self.scale = scale*20
        self.population = []
        for i in range(self.population_size):
            # self.population.append( splprep( knots,k=degree)[0])
            self.population.append( (0, knots.copy(), 0))
            self.mutate(i,True)
        self.scale = scale
            
    def fit(self,pop_size = 10, epochs = 10):
        history = []
        for e in range(self.epochs):
            scores = []
            for i in range(self.population_size):
                    scores.append( (i,self.score(i)) )
            scores.sort(key=lambda a: a[1])
            print(scores[:1])
            
            new_population = []
            history.append(self.population[scores[0][0]][1].copy())
            [ind,scoresval] = zip(*scores)
            
            p = scoresval/np.sum(scoresval)
            ranking = np.cumsum(p)
            
            if e != epochs-1:
                for i in range(self.population_size):
                    partner1 = np.searchsorted(ranking,np.random.uniform(0,1))
                    partner2 = np.searchsorted(ranking,np.random.uniform(0,1))
                    
                    new_population.append(self.cross(ind[partner1],ind[partner2]))
                self.population = new_population.copy()
           
                
        self.scores=scores
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
            tensors = self.model(spline)
        spline = spline.T
        dx_i = [np.matmul(np.matmul( np.array(spline[i]-spline[i+1]).T,np.array(tensors[i]) ),np.array(spline[i]-spline[i+1]) ) for i in range(len(spline)-1)]
        lenght = np.sum(np.sqrt(dx_i))
        
        return lenght
    
# N = 20
# points = np.array([ [0,float(i)/N] for i in range(N+1) ]).T
# sp = EvolutionaryCurve(points,3,5, None,pop_size=10000,epochs=10,scale=0.01)
# print(sp.score(1))
# sp.mutate(1)
# print(sp.score(1))
# sp.cross(1,0)

# print(sp.score(1))
# history = sp.fit()
# from matplotlib import pyplot as plt
# for p in history:
#     plt.plot(p[0],p[1],alpha = 0.5)
    


# plt.plot(history[-1][0],history[-1][1],c = "r",alpha = 1.)
# plt.show()
# aa = sp.population[0]
# print(sp.population[sp.scores[0][0]][0])
# print(sp.population[sp.scores[0][0]][1])
# print(sp.population[sp.scores[0][0]][2])