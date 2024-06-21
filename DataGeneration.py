import numpy as np

class SphereGenerator:
    def __init__(self, radius = 1) -> None:
        self.radius = radius

        
    def create_sphere(self,resolution=20):
        '''
        create sphere with center (cx, cy, cz) and radius r
        '''
        phi = np.linspace(0, 2*np.pi, 2*resolution)
        theta = np.linspace(0, np.pi, resolution)

        theta, phi = np.meshgrid(theta, phi)
        theta= np.reshape(theta,(-1))
        phi = np.reshape(phi,(-1))

        return np.stack([theta,phi]).T
    def distances(self,points):
        distances = []
        pairOfPoints = []
        for p1 in points:
            for p2 in points:
                dist = self.points2dist(p1,p2)
                distances.append(dist)
                pairOfPoints.append(np.array([p1,p2]).reshape((-1)))
        return (pairOfPoints, np.array(distances))

    def points2dist(self,p1,p2):
        r_xy1 = self.radius*np.sin(p1[0])
        x1 =  np.cos(p1[1]) * r_xy1
        y1 =  np.sin(p1[1]) * r_xy1
        z1 =  self.radius * np.cos(p1[0])
        
        r_xy2 = self.radius*np.sin(p2[0])
        x2 =  np.cos(p2[1]) * r_xy2
        y2 =  np.sin(p2[1]) * r_xy2
        z2 =  self.radius * np.cos(p2[0])
        
        dx = x1-x2
        dy = y1-y2
        dz = z1-z2
        
        c_d = np.sqrt(dx**2 + dy**2 + dz**2)
        
        c_a = np.arcsin( 0.5*c_d/self.radius)
        # dist= (self.radius*c_a)**2
        dist = (p1[0]-p2[0])**2 +(p1[1]-p2[1])**2
        return dist
def main():
    sg = SphereGenerator(1)
    points = sg.create_sphere(10)
    pointsPairs,dists = sg.distances(points)
    dx=0.1
    px=np.arcsin(0.9)
    py=7
    print(  sg.points2dist([px,py],[px,py])  )
    print( 0.5*(sg.points2dist([px,py],[px,py+2*dx]) - 2*sg.points2dist([px,py],[px,py+dx])+sg.points2dist([px,py],[px,py]))/dx**2  )
    print( 0.5*(sg.points2dist([px,py],[px+2*dx,py]) - 2*sg.points2dist([px,py],[px+dx,py])+sg.points2dist([px,py],[px,py]))/dx**2  )
    print( 2*(sg.points2dist([px,py],[px+dx,py+dx]) - sg.points2dist([px,py],[px+dx,py])- sg.points2dist([px,py],[px,py+dx]) +sg.points2dist([px,py],[px,py]))/dx*np.sqrt(2)  )

if __name__ == "__main__":
    main()