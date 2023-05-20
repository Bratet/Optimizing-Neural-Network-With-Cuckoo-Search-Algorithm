import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from models import train_nn, predict_nn, cuckoo_search, feedforward, decode_solution
from utils import accuracy_score
from sklearn.model_selection import KFold



# Load and preprocess the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

# Cuckoo Search parameters
n_pop = 250
n_iterations = 1000
pa = 0.5
n_inputs = X.shape[1]
n_hidden = 10
n_outputs = len(np.unique(y))
solution_size = n_inputs * n_hidden + n_hidden * n_outputs + n_hidden + n_outputs

# Train the neural network using Cuckoo Search
kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(X)
cuckoo_scores = []
backprop_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_true_train, y_true_test = y[train_index], y[test_index]
    # Train the neural network
    W1, b1, W2, b2 = train_nn(X_train, y_true_train)
    # Test the trained neural network
    y_pred_train = predict_nn(X_train, W1, b1, W2, b2)
    y_pred_test = predict_nn(X_test, W1, b1, W2, b2)

    train_accuracy = accuracy_score(y_true_train, y_pred_train)
    test_accuracy = accuracy_score(y_true_test, y_pred_test)
    backprop_scores.append(test_accuracy)
    
    
    # Train the neural network using Cuckoo Search
    best_solution = cuckoo_search(X_train, y_true_train, n_pop, n_iterations, pa, n_inputs, n_hidden, n_outputs, solution_size)

    # Apply the best solution to the neural network
    weights, biases = decode_solution(best_solution, n_inputs, n_hidden, n_outputs)

    # Test the trained neural network
    output_train = feedforward(X_train, weights, biases)
    output_test = feedforward(X_test, weights, biases)
    y_pred_train = np.argmax(output_train, axis=1)
    y_pred_test = np.argmax(output_test, axis=1)

    # Calculate the accuracy of the neural network
    train_accuracy = accuracy_score(y_true_train, y_pred_train)
    test_accuracy = accuracy_score(y_true_test, y_pred_test)
    cuckoo_scores.append(test_accuracy)

# calculate the mean and standard deviation of the scores
cuckoo_mean = np.mean(cuckoo_scores)
cuckoo_std = np.std(cuckoo_scores)
backprop_mean = np.mean(backprop_scores)
backprop_std = np.std(backprop_scores)

# print the results of the cuckoo search performance to the normal backpropagation
print('Cuckoo Search: %.3f%% (%.3f)' % (cuckoo_mean, cuckoo_std))
print('Backpropagation: %.3f%% (%.3f)' % (backprop_mean, backprop_std))