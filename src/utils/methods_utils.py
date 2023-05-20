import numpy as np

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the feedforward function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the accuracy score function
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

# Define the accuracy score function
def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
