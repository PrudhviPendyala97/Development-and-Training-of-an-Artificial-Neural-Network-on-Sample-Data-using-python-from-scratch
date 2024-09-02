import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Function to normalize the data
def normalize_data(data):
    normalized_data = data
    for column in range(normalized_data.shape[1]):
        min_val = np.min(normalized_data[:, column])
        max_val = np.max(normalized_data[:, column])
        normalized_data[:, column] = (normalized_data[:, column] - min_val) / (max_val - min_val)
    return normalized_data

# Read the input data from a text file
input_file = "input_data.txt"
data = np.genfromtxt(input_file, delimiter=' ')

# input size based on the number of features
input_size = data.shape[1] - 1

# Initialize other network parameters
hidden_size = 10000
output_size = 1

# Regularization parameter (adjust as needed)
l2_regularization = 0.01

# Randomly initialize weights and biases
np.random.seed(0)
input_weights = np.random.rand(input_size, hidden_size)
hidden_weights = np.random.rand(hidden_size, output_size)
bias_hidden = np.random.rand(1, hidden_size)
bias_output = np.random.rand(1, output_size)

# Normalize the data
normalized_data = normalize_data(data)

X = normalized_data[:, :-1]
y = normalized_data[:, -1]

# Training parameters
learning_rate = 0.4
iteration_count = 500

# Lists to store error values for plotting
error_list = []

# Training the neural network
for iteration in range(iteration_count):
    # Forward propagation
    hidden_input = np.dot(X, input_weights) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, hidden_weights) + bias_output
    output = sigmoid(output_input)

    # Calculate mean squared error
    error = y.reshape(-1, 1) - output
    mse = np.mean(error ** 2)

    # Apply L2 regularization to the weights (excluding biases)
    regularization_term = 0.5 * l2_regularization * (np.sum(input_weights**2) + np.sum(hidden_weights**2))
    mse += regularization_term

    # Backpropagation
    delta_output = error * d_sigmoid(output)
    error_hidden = delta_output.dot(hidden_weights.T)
    delta_hidden = error_hidden * d_sigmoid(hidden_output)

    # Update weights and biases with L2 regularization
    hidden_weights += (np.dot(hidden_output.T, delta_output) - l2_regularization * hidden_weights) * learning_rate
    input_weights += (np.dot(X.T, delta_hidden) - l2_regularization * input_weights) * learning_rate
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

    error_list.append(mse)

# Plot the error versus iteration
plt.plot(range(iteration_count), error_list)
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Error vs Iteration')
plt.grid(True)
plt.show()

# Testing
for i in range(len(X)):
    hidden_input = np.dot(X[i], input_weights) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, hidden_weights) + bias_output
    prediction = sigmoid(output_input)

    print(f"Input: {X[i]}, Target: {y[i]}, Prediction: {prediction}")
