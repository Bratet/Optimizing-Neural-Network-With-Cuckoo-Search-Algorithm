import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from models import train_nn, predict_nn, cuckoo_search, feedforward, decode_solution
from utils import accuracy_score, train_test_split

# Load and preprocess the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train the neural network
W1, b1, W2, b2 = train_nn(X_train, y_train)

# Test the trained neural network
y_pred_train = predict_nn(X_train, W1, b1, W2, b2)
y_pred_test = predict_nn(X_test, W1, b1, W2, b2)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Train accuracy (Backpropagation from scratch as function): {train_accuracy:.2f}")
print(f"Test accuracy (Backpropagation from scratch as function): {test_accuracy:.2f}")
# Cuckoo Search parameters
n_pop = 20
n_iterations = 500
pa = 0.25
n_inputs = X_train.shape[1]
n_hidden = 10
n_outputs = len(np.unique(y))
solution_size = n_inputs * n_hidden + n_hidden * n_outputs + n_hidden + n_outputs

# Train the neural network using Cuckoo Search
best_solution = cuckoo_search(X_train, y_train, n_pop, n_iterations, pa, n_inputs, n_hidden, n_outputs, solution_size)

# Apply the best solution to the neural network
weights, biases = decode_solution(best_solution, n_inputs, n_hidden, n_outputs)

# Test the trained neural network
output_train = feedforward(X_train, weights, biases)
output_test = feedforward(X_test, weights, biases)
y_pred_train = np.argmax(output_train, axis=1)
y_pred_test = np.argmax(output_test, axis=1)

# Calculate the accuracy of the neural network
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Train accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")