import numpy as np
from scipy.optimize import minimize

# Example data for X and Y matrices (8x2 matrices)
# np.random.seed(0)
# X = np.random.rand(8, 2)
# Y = np.random.rand(8, 2)
# X = np.array([[1.00703733, 0.226733482],
#               [0.946614194, -0.569071295],
#               [0.278427903, -0.875027967],
#               [0.544479416,	0.721660537],
#               [-0.256283529, 0.828669238],
#               [-0.912379396, 0.46314663],
#               [-1.013147693, -0.362901165],
#               [-0.511748743, -0.814920671]])

X = np.array([[0.940145648377, 0.23198737889],
              [0.854975266015, -0.527891087187],
              [0.30207685239, -0.881076654125],
              [0.561541247486,	0.723167075326],
              [-0.236100371923, 0.827150010825],
              [-0.89581754248, 0.463599028274],
              [-1.00293262296, -0.364431608943],
              [-0.488763081837, -0.816052896829]])

Y = np.array([[0.75, 0.375],
              [0.75, -0.375],
              [0.375, -0.75],
              [0.375, 0.75],
              [-0.375, 0.75],
              [-0.75, 0.375],
              [-0.75, -0.375],
              [-0.375, -0.75]])

def transform_and_calculate_distance(params, X, Y):
    """
    Transform the matrix X by rotating and translating, then calculate the sum of Euclidean distances to Y.

    :param params: Tuple containing (x_0, y_0, theta_0)
    :param X: Original matrix X
    :param Y: Matrix Y
    :return: Sum of Euclidean distances between transformed X and Y
    """
    x_0, y_0, theta_0 = params
    # Create rotation matrix
    theta_0 = np.deg2rad(theta_0)
    rotation_matrix = np.array([
        [np.cos(theta_0), -np.sin(theta_0)],
        [np.sin(theta_0), np.cos(theta_0)]
    ])
    
    # Apply rotation and translation
    transformed_X = (X @ rotation_matrix.T) + np.array([x_0, y_0])

    # Calculate the sum of Euclidean distances
    distances = np.sqrt(np.sum((transformed_X - Y) ** 2, axis=1))
    return np.sum(distances)

def transform_X(params, X):
    """
    Transform the matrix X by rotating and translating.

    :param params: Tuple containing (x_0, y_0, theta_0)
    :param X: Original matrix X
    """
    x_0, y_0, theta_0 = params
    # Create rotation matrix
    theta_0 = np.deg2rad(theta_0)
    rotation_matrix = np.array([
        [np.cos(theta_0), -np.sin(theta_0)],
        [np.sin(theta_0), np.cos(theta_0)]
    ])
    
    # Apply rotation and translation
    transformed_X = (X @ rotation_matrix.T) + np.array([x_0, y_0])
    return transformed_X

# Initial guess for parameters (x_0, y_0, theta_0)
initial_params = (-0.375, -0.375, 0)

# Perform optimization to minimize the distance sum
result = minimize(transform_and_calculate_distance, initial_params, args=(X, Y), method="Nelder-Mead")

# print(result)
print(result.x)  # Optimized parameters (x_0, y_0, theta_0)
print(transform_and_calculate_distance(list(result.x), X, Y)/8)