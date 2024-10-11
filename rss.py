import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(beta):
    x_1 = np.random.random([1000, beta.shape[0] - 1]) *10
    epsilon = (np.random.rand(1000) -0.5 ) *3
    x = np.column_stack((x_1, np.ones(x_1.shape[0]).T)) 
    y = x @ beta + epsilon.T
    return x, y

def optimize_coeffecients(x, y):
    b = np.linalg.inv(x.T @ x ) @ x.T @ y
    return b

def set_line(b, points):
    x = np.arange(0,11)
    y = b[0] * x + b[1]
    plt.plot(x,y, color = 'red')
    plt.scatter(points[0][:,0], points[1])
    plt.show()


beta = np.array([3,5,2,9,1])
x, y = generate_synthetic_data(beta)
print(optimize_coeffecients(x,y))
