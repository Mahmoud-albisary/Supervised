import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(b_1 = 3, b_0 = 5):
    x_1 = np.random.rand(1000)
    epsilon = (np.random.rand(1000) -0.5 ) *3
    x = np.column_stack((np.transpose(x_1 * 10), np.ones(x_1.shape).T)) 
    y = b_1 * x[:,0] +(b_0 * np.ones(1000)).T + epsilon.T
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

x,y = generate_synthetic_data()
b = optimize_coeffecients(x, y)
set_line(b, (x,y))