import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg

class Generators:
    @staticmethod
    def generate_ndim_sphere(n_dim,n_points,radius,c=None):
        if c is None:
            c = [0]*n_dim
        points = np.random.uniform(-10,10,(n_points,n_dim))
        points /= np.sqrt((points ** 2).sum(-1))[..., np.newaxis]
        return points*radius+c

    @staticmethod
    def generate_ndim_torus(n_dim,n_points,radius,c=None):
        point  = [[0]*n_points]*n_dim
        if c is None:
            c = [0]*n_dim

        tmp = np.random.uniform(-1,1,(n_points,2))
        tmp /= np.sqrt((tmp ** 2).sum(-1))[..., np.newaxis]
        
        lastvector = np.copy(tmp)
        
        point[0] += tmp[:,0]*radius[0]
        point[0+1] += tmp[:,1]*radius[0]
            
        for i in range(1,n_dim-1):
            tmp = np.random.uniform(-1,1,(n_points,2))
            tmp /= np.sqrt((tmp ** 2).sum(-1))[..., np.newaxis]
        
            xx = tmp[:,0]
            x= lastvector*(xx[:,np.newaxis])
            
            for j in np.random.randint(0,n_points,10):
                if all(x[j] == tmp[j,0]*lastvector[j]):
                    print("True")
            lastvector = np.vstack([x.T,tmp.T[1]]).T
            # x = tmp[:,0]
            # xx = np.prod([lastvector[:,0],tmp[:,0]],axis=0)*radius[i]
            # xxx = np.prod([lastvector[:,1],tmp[:,0]],axis=0)*radius[i]
            # point[:i] -= x.T[:i]
            point[:i+1] -= x.T[:i+1]*radius[i]
            point[i+1] += tmp[:,1]*radius[i]
        return np.array(point).T+c




def main():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    p=Generators.generate_ndim_torus(3,4600,[2,1,0.5,0.25,0.12,0.06],[1,0,0])
    ax.scatter(p.T[0], p.T[1],  p.T[2])

    # p=Generators.generate_ndim_sphere(4,600,2)
    # ax.scatter(p.T[0], p.T[1],  p.T[2])

    ax.set_xlim([-2.3,2.3])
    ax.set_ylim([-2.3,2.3])
    ax.set_zlim([-2.3,2.3])
    plt.show()
if __name__ == "__main__":
    main()