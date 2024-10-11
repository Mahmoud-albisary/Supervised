import numpy as np

def generate_synthetic_data(beta):
    x_1 = np.random.random([1000, beta.shape[0] - 1]) *10
    epsilon = (np.random.rand(1000) -0.5 ) *3
    x = np.column_stack((x_1, np.ones(x_1.shape[0]).T)) 
    y = x @ beta + epsilon.T
    return x, y

def optimize_coeffecients(x, y):
    b = np.linalg.inv(x.T @ x ) @ x.T @ y
    return b


beta = np.array([3,5,2,9,1])
x, y = generate_synthetic_data(beta)
print(optimize_coeffecients(x,y))
