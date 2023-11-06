import matplotlib.pyplot as plt
import numpy as np
 

def plotClassifier(X_test, y_test, model, class1, class2, x1Name, x2Name):
    

    class1Points = []
    class2Points = []
    for x, y in zip(X_test, y_test):
        if y == 0:
            class1Points.append(x)
        else:
            class2Points.append(x)

    class1Points = np.array(class1Points)
    class2Points = np.array(class2Points)
    
    plt.scatter(class1Points[:, 0], class1Points[:, 1], color='red', label=class1)
    plt.scatter(class2Points[:, 0], class2Points[:, 1], color='blue', label=class2)
    


    if model.hasBias:
        X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        
    xmin, xmax = min(X_test[:, 1]), max(X_test[:, 1])
    
    xd = np.array([xmin, xmax])

    weights = model.weights
    c = - weights[0] / weights[2]
    m = -weights[1]/weights[2]
    yd = m*xd + c
    plt.plot(xd, yd, 'k', lw=1, ls='--', label="boundary line")


    plt.xlabel(x1Name)
    plt.ylabel(x2Name)
    plt.legend(loc="upper left")
    plt.show()