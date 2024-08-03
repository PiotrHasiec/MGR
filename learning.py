from DataGeneration import DataGenerator
import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
def main():
    sg = DataGenerator(1)
    points = sg.create_sphere(15)
    pointsPairs,dists = sg.distances(points)
    pointsPairs = np.array(pointsPairs)
    
    X_train, X_test, y_train, y_test = train_test_split(pointsPairs,dists,test_size=0.2,random_state=3217)
    
    model = keras.Sequential()
    model.add(keras.Input(shape=(4,)))
    model.add(keras.layers.Dense(16*4*1,"tanh"))
    
    model.add(keras.layers.Dense(8*4*1,"tanh"))
    
    model.add(keras.layers.Dense(4*4*1,"tanh"))
    model.add(keras.layers.Dense(1,"relu"))

    model.compile(optimizer="adam",loss="mae")
    x = X_test[0].reshape((1,-1))
    model.fit(X_train,  y_train,128,20,validation_data = (X_test, y_test))
    
    # with tf.GradientTape() as t2:
    #      with tf.GradientTape() as t3:
    #       with tf.GradientTape() as t1:
            
    #         x1 = np.array([1.,0.]).reshape((1,2))
    #         x2 = np.array([1.,0.]).reshape((1,2))
    #         x1 = tf.convert_to_tensor(x1)
    #         x2 = tf.convert_to_tensor(x2)
    #         t1.watch(x1)
    #         t2.watch(x2)
    #         t3.watch(x2)
    #         x=tf.concat([x1,x2],axis=1)
            
            
    dx=0.1
    px=np.arcsin(0.1)
    py=0
    print(  model(np.array([px,py,px,py]).reshape(1,4))  )
    # print(model(np.array([px,py,px,py+2*dx]).reshape(1,4)))
    print( 2*(model(np.array([px,py,px,py+2*dx]).reshape(1,4)) - 2*model(np.array([px,py,px,py+dx]).reshape(1,4))+model(np.array([px,py,px,py]).reshape(1,4)))/dx**2  )
    # print(model(np.array([px,py,px+2*dx,py]).reshape(1,4)))
    print( 2*(model(np.array([px,py,px+2*dx,py]).reshape(1,4)) - 2*model(np.array([px,py,px+dx,py]).reshape(1,4))+model(np.array([px,py,px,py]).reshape(1,4)))/dx**2  )
    # print(model(np.array([px,py,px+dx,py+dx]).reshape(1,4)))
    print( 2*(model(np.array([px,py,px+dx,py+dx]).reshape(1,4)) - model(np.array([px,py,px+dx,py]).reshape(1,4))- model(np.array([px,py,px,py+dx]).reshape(1,4)) +model(np.array([px,py,px,py]).reshape(1,4)))/dx*np.sqrt(2)  )
    
    test2plot = []
    distances= []
    for i in np.linspace(0,2*np.pi,400):
        test2plot.append(np.array([0,0,i/3,i/8]))
        distances.append(sg.points2dist((0,0),(i/3,i/8)))
    test2plot=np.array(test2plot)
    predictdistances = model(test2plot)
    plt.plot(np.linspace(0,2*np.pi,400),distances, label = "True")
    plt.plot(np.linspace(0,2*np.pi,400),predictdistances, label = "Predicted")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()