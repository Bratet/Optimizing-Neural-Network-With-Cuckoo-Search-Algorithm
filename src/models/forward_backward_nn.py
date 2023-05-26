import numpy as np

# Sigmoid function - a common activation function used in neural networks, squashes values to range [0, 1]
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function - used in the backpropagation process during training
def sigmoid_derivative(x):
    return x * (1 - x)

# One-hot encoding function - converts class labels into a binary vector representation
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def train_nn(X_train, y_train, n_hidden=10, learning_rate=0.01, epochs=500):
    # Determine the dimensions of the input data and the number of unique classes
    n_inputs = X_train.shape[1]
    n_outputs = len(np.unique(y_train))
    n_samples = X_train.shape[0]

    # Initialize random weights and biases for the layers of the neural network
    W1 = np.random.randn(n_inputs, n_hidden)
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, n_outputs)
    b2 = np.zeros((1, n_outputs))

    # Convert the labels into one-hot encoded format
    y_train_encoded = one_hot_encode(y_train, n_outputs)

    # Create a list to store the loss at each epoch
    loss_history = []

    # Training loop
    for _ in range(epochs):
        # Forward propagation: compute the output of each layer given the previous layer's output
        hidden_layer = sigmoid(np.dot(X_train, W1) + b1)
        output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)

        # Compute the error at the output
        output_error = y_train_encoded - output_layer

        # Compute loss (Mean Squared Error) and append to loss history
        loss = np.mean(np.square(output_error))
        loss_history.append(loss)

        # Backward propagation: compute deltas
        output_delta = output_error * sigmoid_derivative(output_layer)
        hidden_error = np.dot(output_delta, W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)

        # Update weights and biases
        W2 += learning_rate * np.dot(hidden_layer.T, output_delta)
        b2 += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        W1 += learning_rate * np.dot(X_train.T, hidden_delta)
        b1 += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    # Return the trained parameters and the loss history
    return W1, b1, W2, b2, loss_history

# Function to make predictions given a trained model (weights and biases) and input data
def predict_nn(X, W1, b1, W2, b2):
    # Compute output of each layer (similar to the forward propagation step during training)
    hidden_layer = sigmoid(np.dot(X, W1) + b1)
    output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)
    
    # Return the class with the highest probability from the output layer
    return np.argmax(output_layer, axis=1)

