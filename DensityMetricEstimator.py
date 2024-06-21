import numpy as np
from scipy import integrate as integrate
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
