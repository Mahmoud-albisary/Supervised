import numpy as np

def generate_synthetic_data(beta):
    x_1 = np.random.random([1000, beta.shape[0] - 1]) *10
    epsilon = (np.random.rand(1000) -0.5 ) *3
    x = np.column_stack((x_1, np.ones(x_1.shape[0]).T)) 
    y = x @ beta + epsilon.T
    return x, y

def euclidean_distance(x_new, x):
    return np.linalg.norm(x_new - x, axis = 1)

def knn_algorithm(x, new_x, y, k =1):
    dist = euclidean_distance(new_x, x)
    nearest_indices = np.argsort(dist)[:k]
    nearest_y = y[nearest_indices]
    return np.mean(nearest_y)

beta = np.array([3,5,2,9,1])
new_x = np.array([4,2,7,3,5])
X, y = generate_synthetic_data(beta)
new_y = knn_algorithm(X, new_x, y, k = 1)
print(new_y)

    