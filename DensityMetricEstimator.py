import numpy as np
from scipy import integrate as integrate
from sklearn.neighbors import NearestNeighbors
class DensityMEtricEstimator:
    def __init__(self, kde) -> None:
        self.kde = kde
        
    def calculate_dist(self,x,y):
        normal_dist = np.linalg.norm(y-x)
        
        curve = lambda t: x + (y-x)*t
        t = np.linspace(0,1,20,endpoint=True)
        points =  [curve(ti) for ti in t]
        # det = np.exp(self.kde.score_samples(points)*(1./float(x.shape[0]))  )
        det = np.exp(self.kde.score_samples(points)*(1./float(x.shape[0]))) 
        # det = np.where(abs(det) == np.inf, 1000,det)

        normal_dist = integrate.trapezoid(det,dx = t[1]-t[0])*normal_dist
        return normal_dist
    def dist_tab(self,x):
        self.knn = NearestNeighbors(n_neighbors=5)
        self.knn.fit(x)
        x = np.array(x)
        x_nn_ind = self.knn.kneighbors(x,return_distance=False)
        tab = np.zeros( (x.shape[0],x.shape[0]))
        for i,xi in enumerate(x):
            for j in x[i]:
                if(i<j):
                    tab[i][j]=self.calculate_dist(xi,x[j])
        return tab
            
