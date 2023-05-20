import numpy as np

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the accuracy score function
def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Define the feedforward function
def feedforward(X, weights, biases):
    hidden_layer = sigmoid(np.dot(X, weights[0]) + biases[0])
    output_layer = sigmoid(np.dot(hidden_layer, weights[1]) + biases[1])
    return output_layer

# Define the decode solution function
def decode_solution(solution, n_inputs, n_hidden, n_outputs):
    weights = []
    biases = []

    w1_size = n_inputs * n_hidden
    w2_size = n_hidden * n_outputs

    w1 = solution[:w1_size].reshape((n_inputs, n_hidden))
    w2 = solution[w1_size:w1_size + w2_size].reshape((n_hidden, n_outputs))
    b1 = solution[w1_size + w2_size:w1_size + w2_size + n_hidden]
    b2 = solution[w1_size + w2_size + n_hidden:]

    weights.append(w1)
    weights.append(w2)
    biases.append(b1)
    biases.append(b2)

    return weights, biases

# Define the fitness function
def fitness(solution, X, y, n_inputs, n_hidden, n_outputs):
    weights, biases = decode_solution(solution, n_inputs, n_hidden, n_outputs)
    output = feedforward(X, weights, biases)
    y_pred = np.argmax(output, axis=1)
    return -accuracy_score(y, y_pred)

def cuckoo_search(X, y, n_pop, n_iterations, pa, n_inputs, n_hidden, n_outputs, solution_size):
    # Initialize population and fitness
    population = np.random.uniform(-1, 1, (n_pop, solution_size))
    fitnesses = np.array([fitness(sol, X, y, n_inputs, n_hidden, n_outputs) for sol in population])
    
    # Find the best solution and its fitness
    min_idx = np.argmin(fitnesses)
    best_solution = population[min_idx]
    best_fitness = fitnesses[min_idx]

    for _ in range(n_iterations):
        # Generate a cuckoo solution
        cuckoo_idx = np.random.randint(n_pop)
        step_size = np.random.normal(0, 0.1, solution_size)
        cuckoo = population[cuckoo_idx] + step_size

        # Evaluate the fitness of the cuckoo solution
        cuckoo_fitness = fitness(cuckoo, X, y, n_inputs, n_hidden, n_outputs)

        # Select a random solution from the population
        random_idx = np.random.randint(n_pop)

        # Replace the random solution with the cuckoo solution if it has better fitness
        if cuckoo_fitness < fitnesses[random_idx]:
            population[random_idx] = cuckoo
            fitnesses[random_idx] = cuckoo_fitness

        # Replace a fraction pa of the worst solutions with new randomly generated solutions
        n_replace = int(n_pop * pa)
        worst_indices = np.argsort(fitnesses)[-n_replace:]
        population[worst_indices] = np.random.uniform(-1, 1, (n_replace, solution_size))
        fitnesses[worst_indices] = np.array([fitness(sol, X, y, n_inputs, n_hidden, n_outputs) for sol in population[worst_indices]])

        # Update the best solution found so far
        min_idx = np.argmin(fitnesses)
        current_best_solution = population[min_idx]
        current_best_fitness = fitnesses[min_idx]

        if current_best_fitness < best_fitness:
            best_solution = current_best_solution
            best_fitness = current_best_fitness

    return best_solution
