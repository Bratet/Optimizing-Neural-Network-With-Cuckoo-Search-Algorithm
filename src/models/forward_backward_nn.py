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

def train_nn(X_train, y_train, n_hidden=10, learning_rate=0.01, epochs=500):
    n_inputs = X_train.shape[1]
    n_outputs = len(np.unique(y_train))
    n_samples = X_train.shape[0]

    W1 = np.random.randn(n_inputs, n_hidden)
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, n_outputs)
    b2 = np.zeros((1, n_outputs))

    y_train_encoded = one_hot_encode(y_train, n_outputs)

    for _ in range(epochs):
        hidden_layer = sigmoid(np.dot(X_train, W1) + b1)
        output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)

        output_error = y_train_encoded - output_layer
        output_delta = output_error * sigmoid_derivative(output_layer)

        hidden_error = np.dot(output_delta, W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)

        W2 += learning_rate * np.dot(hidden_layer.T, output_delta)
        b2 += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        W1 += learning_rate * np.dot(X_train.T, hidden_delta)
        b1 += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    return W1, b1, W2, b2

def predict_nn(X, W1, b1, W2, b2):
    hidden_layer = sigmoid(np.dot(X, W1) + b1)
    output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)
    return np.argmax(output_layer, axis=1)
