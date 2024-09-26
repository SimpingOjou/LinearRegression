import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def computeCost(X, y, theta):
    '''
        Compute the cost function.
    '''

    tobesummed = np.power(((X @ theta.T) - y), 2) # @ is the dot product operator

    return np.sum(tobesummed) / (2 * len(X))

def gradientDescent(X, y, theta, iters, alpha):
    '''
        Run the gradient descent algorithm.
    '''

    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    
    return theta, cost

if __name__ == '__main__':
    my_data = pd.read_csv('./src/home.csv',names=["size (m^2)","bedroom (nbr)","price (Eur)"])

    print(my_data.head())

    # Normalize the data to evenly distribute the values
    my_data = (my_data - np.mean(my_data, axis=0)) / np.std(my_data, axis=0)
    print(my_data.head())

    # Setting up the matrixes
    X = my_data.iloc[:,0:2]
    ones = np.ones([X.shape[0],1])
    X = np.concatenate((ones,X),axis=1)

    y = my_data.iloc[:,2:3].values # Converts it from pandas.core.frame.DataFrame to numpy.ndarray
    theta = np.zeros([1,3])

    # Set hyper parameters
    alpha = 0.01
    iters = 1000

    # Running the gd and cost function
    g, cost = gradientDescent(X,y,theta,iters,alpha)

    finalCost = computeCost(X,y,g)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Assuming X has at least 3 columns
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])

    ax.set_xlabel("size (m^2)")
    ax.set_ylabel("bedroom (nbr)")
    ax.set_zlabel("price (Eur)")

    plt.figure()
    plt.plot(np.arange(iters), cost, label='Error')  
    plt.xlabel('Iterations')  
    plt.ylabel('Cost')  
    plt.title('Error over iterations')  
    plt.legend()
    plt.grid()

    plt.show()



