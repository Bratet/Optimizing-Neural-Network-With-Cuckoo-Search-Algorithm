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

# define train_test_split function to split data into train and test sets
def train_test_split(X, y, test_size=0.2):
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]