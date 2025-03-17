#importing packages
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import pandas as pd
import numpy as np
import wandb
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split


# Function to preprocess 
def preprocess_image_data(data):
    return np.array([i.flatten() / 255.0 for i in data])

class NeuralNetwork:
    def __init__(self, input_neurons, output_neurons, config):

        self.hidden_layers = config["hidden_layers"]
        self.hidden_neurons = config["hl_size"]
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.total_layers = self.hidden_layers + 1
        self.output_layer_index = self.total_layers - 1
        self.config = config
        self.input_layer_neurons = input_neurons
        self.output_layer_neurons = output_neurons

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        if self.config["initialization"] == "random":
            self._initialize_random()
        elif self.config["initialization"] == "xavier":
            self._initialize_xavier()

    def _initialize_random(self):
        """Initializes weights and biases with small randomly."""
        for i in range(self.total_layers):
            if i == 0:
                layer_weights = np.random.randn(self.hidden_neurons, self.input_neurons) * 0.01
                layer_biases = np.random.randn(self.hidden_neurons, 1) * 0.01
            elif i == self.output_layer_index:
                layer_weights = np.random.randn(self.output_neurons, self.hidden_neurons) * 0.01
                layer_biases = np.random.randn(self.output_neurons, 1) * 0.01
            else:
                layer_weights = np.random.randn(self.hidden_neurons, self.hidden_neurons) * 0.01
                layer_biases = np.random.randn(self.hidden_neurons, 1) * 0.01
            self.weights.append(layer_weights)
            self.biases.append(layer_biases)

    def _initialize_xavier(self):
        """Initializes weights using Xavier initialization and biases randomly."""
        for i in range(self.total_layers):
            if i == 0:
                scale = np.sqrt(2.0 / (self.hidden_neurons + self.input_neurons))
                layer_weights = np.random.randn(self.hidden_neurons, self.input_neurons) * scale
                layer_biases = np.zeros((self.hidden_neurons, 1))
            elif i == self.output_layer_index:
                scale = np.sqrt(2.0 / (self.hidden_neurons + self.output_neurons))
                layer_weights = np.random.randn(self.output_neurons, self.hidden_neurons) * scale
                layer_biases = np.zeros((self.output_neurons, 1))
            else:
                scale = np.sqrt(2.0 / (self.hidden_neurons + self.hidden_neurons))
                layer_weights = np.random.randn(self.hidden_neurons, self.hidden_neurons) * scale
                layer_biases = np.zeros((self.hidden_neurons, 1))
            self.weights.append(layer_weights)
            self.biases.append(layer_biases)

    # Activation functions and their derivatives
    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


    def tanh(self, x):
        return np.tanh(x)


    def tanh_derivative(self, x):
        return 1 - self.tanh(x) ** 2


    def relu(self, x):
        return np.maximum(0, x)


    def relu_derivative(self, x):
        result = np.zeros_like(x)
        for i in range(len(x)):
            if x[i] > 0:
                result[i] = 1
        return result


    def softmax(self, x):
        log_softmax = x - np.max(x)  # For Numerical stability
        log_softmax -= np.log(np.sum(np.exp(log_softmax), axis=0))
        return np.exp(log_softmax)


    def apply_activation(self, activation_name, x):
        activation_functions = {
            "sigmoid": self.sigmoid,
            "relu": self.relu,
            "tanh": self.tanh,
        }
        if activation_name in activation_functions:
            return activation_functions[activation_name](x)


    # Method to compute activation derivative
    def compute_activation_derivative(self, activation_name, x):
        if activation_name == "sigmoid":
            return self.sigmoid_derivative(x)
        elif activation_name == "relu":
            return self.relu_derivative(x)
        elif activation_name == "tanh":
            return self.tanh_derivative(x)

    # Method to perform backpropagation and compute gradients for weights and biases
    def compute_gradients(self, activations, pre_activations, true_label, input_data):
        # Initialize lists to store gradients for activations, pre-activations, weights, and biases
        grad_activations = [None] * self.total_layers
        grad_pre_activations = [None] * self.total_layers
        grad_weights = [None] * self.total_layers
        grad_biases = [None] * self.total_layers

        # Create one-hot encoded vector for the true label
        one_hot_label = np.zeros((self.output_layer_neurons, 1))
        one_hot_label[true_label] = 1

        # Compute the gradient of the loss with respect to the output layer's pre-activations
        if self.config["loss"] == "cross_entropy":
            grad_pre_activations[self.total_layers - 1] = -(one_hot_label - activations[self.total_layers - 1])
        elif self.config["loss"] == "mean_squared_error":
            output_activation = activations[self.total_layers - 1]
            grad_pre_activations[self.total_layers - 1] = (output_activation - one_hot_label) * output_activation * (1 - output_activation)

        # Iterate through layers in reverse order to compute gradients
        for layer_idx in range(self.total_layers - 1, -1, -1):
            if layer_idx == 0:
                # Compute weight gradients for the input layer
                grad_weights[layer_idx] = np.matmul(grad_pre_activations[layer_idx], input_data.reshape(1, -1))
            else:
                # Compute weight gradients for hidden and output layers
                grad_weights[layer_idx] = np.matmul(grad_pre_activations[layer_idx], activations[layer_idx - 1].T)

            # Bias gradients are the same as the pre-activation gradients
            grad_biases[layer_idx] = np.copy(grad_pre_activations[layer_idx])

            if layer_idx - 1 >= 0:
                # Compute gradients for the previous layer's activations
                grad_activations[layer_idx - 1] = np.matmul(self.weights[layer_idx].T, grad_pre_activations[layer_idx])
                # Compute gradients for the previous layer's pre-activations
                grad_pre_activations[layer_idx - 1] = grad_activations[layer_idx - 1] * self.compute_activation_derivative(self.config["activation"], pre_activations[layer_idx - 1])

        # Return the computed gradients for weights and biases
        return grad_weights, grad_biases

    # Takes a flattened image as input and returns activations and pre-activations for all layers
    def forward_propagate(self, input_data):
        # Initialize lists to store pre-activations and activations for each layer
        pre_activations = [None] * self.total_layers
        activations = [None] * self.total_layers

        # Iterate through each layer in the network
        for layer_idx in range(self.total_layers):
            if layer_idx == 0:
                # Compute pre-activation for the input layer
                pre_activations[layer_idx] = np.matmul(self.weights[layer_idx], input_data.reshape(self.input_layer_neurons, 1)) + self.biases[layer_idx]
                # Apply activation function to get the activation for the input layer
                activations[layer_idx] = self.apply_activation(self.config["activation"], pre_activations[layer_idx])

            elif layer_idx == self.total_layers - 1:
                # Compute pre-activation for the output layer
                pre_activations[layer_idx] = np.matmul(self.weights[layer_idx], activations[layer_idx - 1]) + self.biases[layer_idx]
                # Apply softmax activation for the output layer
                activations[layer_idx] = self.softmax(pre_activations[layer_idx])

            else:
                # Compute pre-activation for hidden layers
                pre_activations[layer_idx] = np.matmul(self.weights[layer_idx], activations[layer_idx - 1]) + self.biases[layer_idx]
                # Apply activation function for hidden layers
                activations[layer_idx] = self.apply_activation(self.config["activation"], pre_activations[layer_idx])

        return activations, pre_activations

    def update_parameters_with_momentum(self, weight_updates, bias_updates):
        for layer_index in range(self.total_layers):
            self.weights[layer_index] -= weight_updates[layer_index]
            self.biases[layer_index] -= bias_updates[layer_index]

    # Method to perform gradient descent with momentum optimization
    def momentum_based_gradient_descent(self, training_data, training_labels, validation_data, validation_labels):
        # Initialize previous updates for weights and biases with zeros
        previous_weight_updates = [np.zeros_like(weight) for weight in self.weights]
        previous_bias_updates = [np.zeros_like(bias) for bias in self.biases]

        # Temporary variables to store current updates
        current_weight_updates = [np.zeros_like(weight) for weight in self.weights]
        current_bias_updates = [np.zeros_like(bias) for bias in self.biases]

        # Hyperparameters
        momentum_beta = self.config["momentum_beta"]
        batch_size = self.config["batch_size"]
        learning_rate = self.config["learning_rate"]
        weight_decay = self.config["weight_decay"]

        # Training loop over epochs
        for epoch in range(self.config["epochs"]):
            # Process data in mini-batches
            for batch_start in range(0, len(training_data), batch_size):
                batch_data = training_data[batch_start:batch_start + batch_size]
                batch_labels = training_labels[batch_start:batch_start + batch_size]

                # Initialize gradients for weights and biases
                weight_gradients = [np.zeros_like(weight) for weight in self.weights]
                bias_gradients = [np.zeros_like(bias) for bias in self.biases]

                # Compute gradients for each sample in the batch
                for sample_index in range(len(batch_data)):
                    activations, pre_activations = self.forward_propagate(batch_data[sample_index])
                    sample_weight_gradients, sample_bias_gradients = self.compute_gradients(activations, pre_activations, batch_labels[sample_index], batch_data[sample_index])

                    # Accumulate gradients
                    for layer_index in range(self.total_layers):
                        weight_gradients[layer_index] += sample_weight_gradients[layer_index]
                        bias_gradients[layer_index] += sample_bias_gradients[layer_index]

                # Update weights and biases using momentum
                for layer_index in range(self.total_layers):
                    # Update weights
                    current_weight_updates[layer_index] = momentum_beta * previous_weight_updates[layer_index] + learning_rate * weight_gradients[layer_index]
                    self.weights[layer_index] -= current_weight_updates[layer_index] + weight_decay * self.weights[layer_index]
                    previous_weight_updates[layer_index] = current_weight_updates[layer_index]

                    # Update biases
                    current_bias_updates[layer_index] = momentum_beta * previous_bias_updates[layer_index] + learning_rate * bias_gradients[layer_index]
                    self.biases[layer_index] -= current_bias_updates[layer_index]
                    previous_bias_updates[layer_index] = current_bias_updates[layer_index]

            # Calculate and log loss at specific intervals
            if (self.config["epochs"] == 10 and epoch % 2 == 1) or self.config["epochs"] == 5:
                self.calculate_loss(training_data, training_labels, validation_data, validation_labels, epoch)

    # Method to perform stochastic gradient descent with mini-batches
    def stochastic_gradient_descent(self, training_data, training_labels, validation_data, validation_labels):
        # Hyperparameters
        batch_size = self.config["batch_size"]
        learning_rate = self.config["learning_rate"]
        weight_decay = self.config["weight_decay"]

        # Training loop over epochs
        for epoch in range(self.config["epochs"]):
            # Process data in mini-batches
            for batch_start in range(0, len(training_data), batch_size):
                batch_data = training_data[batch_start:batch_start + batch_size]
                batch_labels = training_labels[batch_start:batch_start + batch_size]

                # Initialize gradients for weights and biases
                weight_gradients = [np.zeros_like(weight) for weight in self.weights]
                bias_gradients = [np.zeros_like(bias) for bias in self.biases]

                # Compute gradients for each sample in the batch
                for sample_index in range(len(batch_data)):
                    # Forward pass
                    activations, pre_activations = self.forward_propagate(batch_data[sample_index])
                    # Backward pass
                    sample_weight_gradients, sample_bias_gradients = self.compute_gradients(activations, pre_activations, batch_labels[sample_index], batch_data[sample_index])

                    # Accumulate gradients
                    for layer_index in range(self.total_layers):
                        weight_gradients[layer_index] += sample_weight_gradients[layer_index]
                        bias_gradients[layer_index] += sample_bias_gradients[layer_index]

                # Update weights and biases using the computed gradients
                for layer_index in range(self.total_layers):
                    # Update weights with weight decay
                    self.weights[layer_index] -= learning_rate * weight_gradients[layer_index] + weight_decay * self.weights[layer_index]
                    # Update biases
                    self.biases[layer_index] -= learning_rate * bias_gradients[layer_index]

            # Calculate and log loss at specific intervals
            if (self.config["epochs"] == 10 and epoch % 2 == 1) or self.config["epochs"] == 5:
                self.calculate_loss(training_data, training_labels, validation_data, validation_labels, epoch)

    def nesterov_gradient_descent(self, training_data, training_labels, validation_data, validation_labels):
        # Initialize previous updates for weights and biases with zeros
        previous_weight_updates = [np.zeros_like(weight) for weight in self.weights]
        previous_bias_updates = [np.zeros_like(bias) for bias in self.biases]

        # Temporary variables to store current updates
        current_weight_updates = [np.zeros_like(weight) for weight in self.weights]
        current_bias_updates = [np.zeros_like(bias) for bias in self.biases]

        # Hyperparameters
        momentum_beta = self.config["momentum_beta"]
        batch_size = self.config["batch_size"]
        learning_rate = self.config["learning_rate"]
        weight_decay = self.config["weight_decay"]

        # Training loop over epochs
        for epoch in range(self.config["epochs"]):
            # Process data in mini-batches
            for batch_start in range(0, len(training_data), batch_size):
                batch_data = training_data[batch_start:batch_start + batch_size]
                batch_labels = training_labels[batch_start:batch_start + batch_size]

                # Initialize gradients for weights and biases
                weight_gradients = [np.zeros_like(weight) for weight in self.weights]
                bias_gradients = [np.zeros_like(bias) for bias in self.biases]

                # Compute lookahead updates for weights and biases
                for layer_index in range(self.total_layers):
                    current_weight_updates[layer_index] = momentum_beta * previous_weight_updates[layer_index]
                    current_bias_updates[layer_index] = momentum_beta * previous_bias_updates[layer_index]

                # Update parameters temporarily for lookahead gradient calculation
                self.update_parameters_with_momentum(current_weight_updates, current_bias_updates)

                # Compute gradients for each sample in the batch
                for sample_index in range(len(batch_data)):
                    # Forward pass
                    activations, pre_activations = self.forward_propagate(batch_data[sample_index])
                    # Backward pass
                    sample_weight_gradients, sample_bias_gradients = self.compute_gradients(activations, pre_activations, batch_labels[sample_index], batch_data[sample_index])

                    # Accumulate gradients
                    for layer_index in range(self.total_layers):
                        weight_gradients[layer_index] += sample_weight_gradients[layer_index]
                        bias_gradients[layer_index] += sample_bias_gradients[layer_index]

                # Update weights and biases using Nesterov Accelerated Gradient Descent
                for layer_index in range(self.total_layers):
                    # Update weights
                    previous_weight_updates[layer_index] = current_weight_updates[layer_index] + learning_rate * weight_gradients[layer_index]
                    self.weights[layer_index] -= previous_weight_updates[layer_index] + weight_decay * self.weights[layer_index]

                    # Update biases
                    previous_bias_updates[layer_index] = current_bias_updates[layer_index] + learning_rate * bias_gradients[layer_index]
                    self.biases[layer_index] -= previous_bias_updates[layer_index]

            # Calculate and log loss at specific intervals
            if (self.config["epochs"] == 10 and epoch % 2 == 1) or self.config["epochs"] == 5:
                self.calculate_loss(training_data, training_labels, validation_data, validation_labels, epoch)

    # Method to perform RMSProp optimization
    def rmsprop_optimization(self, training_data, training_labels, validation_data, validation_labels):
        # Initialize exponentially weighted averages of squared gradients for weights and biases
        squared_grad_weights = [np.zeros_like(weight) for weight in self.weights]
        squared_grad_biases = [np.zeros_like(bias) for bias in self.biases]

        # Hyperparameters
        decay_rate = self.config["rms_beta"]  # Decay rate for the moving average
        epsilon = 1e-4  # Small constant to avoid division by zero
        batch_size = self.config["batch_size"]
        learning_rate = self.config["learning_rate"]
        weight_decay = self.config["weight_decay"]

        # Training loop over epochs
        for epoch in range(self.config["epochs"]):
            # Process data in mini-batches
            for batch_start in range(0, len(training_data), batch_size):
                batch_data = training_data[batch_start:batch_start + batch_size]
                batch_labels = training_labels[batch_start:batch_start + batch_size]

                # Initialize gradients for weights and biases
                weight_gradients = [np.zeros_like(weight) for weight in self.weights]
                bias_gradients = [np.zeros_like(bias) for bias in self.biases]

                # Compute gradients for each sample in the batch
                for sample_index in range(len(batch_data)):
                    # Forward pass
                    activations, pre_activations = self.forward_propagate(batch_data[sample_index])
                    # Backward pass
                    sample_weight_gradients, sample_bias_gradients = self.compute_gradients(activations, pre_activations, batch_labels[sample_index], batch_data[sample_index])

                    # Accumulate gradients
                    for layer_index in range(self.total_layers):
                        weight_gradients[layer_index] += sample_weight_gradients[layer_index]
                        bias_gradients[layer_index] += sample_bias_gradients[layer_index]

                # Update exponentially weighted averages of squared gradients and parameters
                for layer_index in range(self.total_layers):
                    # Update squared gradients for weights
                    squared_grad_weights[layer_index] = decay_rate * squared_grad_weights[layer_index] + (1 - decay_rate) * (weight_gradients[layer_index] ** 2)
                    # Update weights with RMSProp and weight decay
                    self.weights[layer_index] -= learning_rate * weight_gradients[layer_index] / (np.sqrt(squared_grad_weights[layer_index]) + epsilon) + weight_decay * self.weights[layer_index]

                    # Update squared gradients for biases
                    squared_grad_biases[layer_index] = decay_rate * squared_grad_biases[layer_index] + (1 - decay_rate) * (bias_gradients[layer_index] ** 2)
                    # Update biases with RMSProp
                    self.biases[layer_index] -= learning_rate * bias_gradients[layer_index] / (np.sqrt(squared_grad_biases[layer_index]) + epsilon)

            # Calculate and log loss at specific intervals
            if (self.config["epochs"] == 10 and epoch % 2 == 1) or self.config["epochs"] == 5:
                self.calculate_loss(training_data, training_labels, validation_data, validation_labels, epoch)

    # Method to perform Adam optimization
    def adam_optimization(self, training_data, training_labels, validation_data, validation_labels):
        # Initialize first and second moment estimates for weights and biases
        first_moment_weights = [np.zeros_like(weight) for weight in self.weights]
        first_moment_biases = [np.zeros_like(bias) for bias in self.biases]
        second_moment_weights = [np.zeros_like(weight) for weight in self.weights]
        second_moment_biases = [np.zeros_like(bias) for bias in self.biases]

        # Initialize bias-corrected moment estimates
        first_moment_weights_hat = [np.zeros_like(weight) for weight in self.weights]
        first_moment_biases_hat = [np.zeros_like(bias) for bias in self.biases]
        second_moment_weights_hat = [np.zeros_like(weight) for weight in self.weights]
        second_moment_biases_hat = [np.zeros_like(bias) for bias in self.biases]

        # Hyperparameters
        beta1 = self.config["beta1"]  # Exponential decay rate for the first moment estimates
        beta2 = self.config["beta2"]  # Exponential decay rate for the second moment estimates
        epsilon = self.config["eps"]  # Small constant to avoid division by zero
        batch_size = self.config["batch_size"]
        learning_rate = self.config["learning_rate"]
        weight_decay = self.config["weight_decay"]

        # Training loop over epochs
        for epoch in range(self.config["epochs"]):
            # Process data in mini-batches
            for batch_start in range(0, len(training_data), batch_size):
                batch_data = training_data[batch_start:batch_start + batch_size]
                batch_labels = training_labels[batch_start:batch_start + batch_size]

                # Initialize gradients for weights and biases
                weight_gradients = [np.zeros_like(weight) for weight in self.weights]
                bias_gradients = [np.zeros_like(bias) for bias in self.biases]

                # Compute gradients for each sample in the batch
                for sample_index in range(len(batch_data)):
                    # Forward pass
                    activations, pre_activations = self.forward_propagate(batch_data[sample_index])
                    # Backward pass
                    sample_weight_gradients, sample_bias_gradients = self.compute_gradients(activations, pre_activations, batch_labels[sample_index], batch_data[sample_index])

                    # Accumulate gradients
                    for layer_index in range(self.total_layers):
                        weight_gradients[layer_index] += sample_weight_gradients[layer_index]
                        bias_gradients[layer_index] += sample_bias_gradients[layer_index]

                # Update first and second moment estimates and parameters
                for layer_index in range(self.total_layers):
                    # Update first moment estimates (mean)
                    first_moment_weights[layer_index] = beta1 * first_moment_weights[layer_index] + (1 - beta1) * weight_gradients[layer_index]
                    first_moment_biases[layer_index] = beta1 * first_moment_biases[layer_index] + (1 - beta1) * bias_gradients[layer_index]

                    # Update second moment estimates (uncentered variance)
                    second_moment_weights[layer_index] = beta2 * second_moment_weights[layer_index] + (1 - beta2) * (weight_gradients[layer_index] ** 2)
                    second_moment_biases[layer_index] = beta2 * second_moment_biases[layer_index] + (1 - beta2) * (bias_gradients[layer_index] ** 2)

                    # Compute bias-corrected moment estimates
                    first_moment_weights_hat[layer_index] = first_moment_weights[layer_index] / (1 - np.power(beta1, epoch + 1))
                    first_moment_biases_hat[layer_index] = first_moment_biases[layer_index] / (1 - np.power(beta1, epoch + 1))
                    second_moment_weights_hat[layer_index] = second_moment_weights[layer_index] / (1 - np.power(beta2, epoch + 1))
                    second_moment_biases_hat[layer_index] = second_moment_biases[layer_index] / (1 - np.power(beta2, epoch + 1))

                    # Update weights and biases using Adam optimization
                    self.weights[layer_index] -= learning_rate * first_moment_weights_hat[layer_index] / (np.sqrt(second_moment_weights_hat[layer_index]) + epsilon) + weight_decay * self.weights[layer_index]
                    self.biases[layer_index] -= learning_rate * first_moment_biases_hat[layer_index] / (np.sqrt(second_moment_biases_hat[layer_index]) + epsilon)

            # Calculate and log loss at specific intervals
            if (self.config["epochs"] == 10 and epoch % 2 == 1) or self.config["epochs"] == 5:
                self.calculate_loss(training_data, training_labels, validation_data, validation_labels, epoch)

    # Method to perform NAdam optimization
    def nadam_optimization(self, training_data, training_labels, validation_data, validation_labels):
        # Initialize first and second moment estimates for weights and biases
        first_moment_weights = [np.zeros_like(weight) for weight in self.weights]
        first_moment_biases = [np.zeros_like(bias) for bias in self.biases]
        second_moment_weights = [np.zeros_like(weight) for weight in self.weights]
        second_moment_biases = [np.zeros_like(bias) for bias in self.biases]

        # Initialize bias-corrected moment estimates
        first_moment_weights_hat = [np.zeros_like(weight) for weight in self.weights]
        first_moment_biases_hat = [np.zeros_like(bias) for bias in self.biases]
        second_moment_weights_hat = [np.zeros_like(weight) for weight in self.weights]
        second_moment_biases_hat = [np.zeros_like(bias) for bias in self.biases]

        # Hyperparameters
        beta1 = self.config["beta1"]  # Exponential decay rate for the first moment estimates
        beta2 = self.config["beta2"]  # Exponential decay rate for the second moment estimates
        epsilon = self.config["eps"]  # Small constant to avoid division by zero
        batch_size = self.config["batch_size"]
        learning_rate = self.config["learning_rate"]
        weight_decay = self.config["weight_decay"]

        # Training loop over epochs
        for epoch in range(self.config["epochs"]):
            # Process data in mini-batches
            for batch_start in range(0, len(training_data), batch_size):
                batch_data = training_data[batch_start:batch_start + batch_size]
                batch_labels = training_labels[batch_start:batch_start + batch_size]

                # Initialize gradients for weights and biases
                weight_gradients = [np.zeros_like(weight) for weight in self.weights]
                bias_gradients = [np.zeros_like(bias) for bias in self.biases]

                # Compute gradients for each sample in the batch
                for sample_index in range(len(batch_data)):
                    # Forward pass
                    activations, pre_activations = self.forward_propagate(batch_data[sample_index])
                    # Backward pass
                    sample_weight_gradients, sample_bias_gradients = self.compute_gradients(activations, pre_activations, batch_labels[sample_index], batch_data[sample_index])

                    # Accumulate gradients
                    for layer_index in range(self.total_layers):
                        weight_gradients[layer_index] += sample_weight_gradients[layer_index]
                        bias_gradients[layer_index] += sample_bias_gradients[layer_index]

                # Update first and second moment estimates and parameters
                for layer_index in range(self.total_layers):
                    # Update first moment estimates (mean)
                    first_moment_weights[layer_index] = beta1 * first_moment_weights[layer_index] + (1 - beta1) * weight_gradients[layer_index]
                    first_moment_biases[layer_index] = beta1 * first_moment_biases[layer_index] + (1 - beta1) * bias_gradients[layer_index]

                    # Update second moment estimates (uncentered variance)
                    second_moment_weights[layer_index] = beta2 * second_moment_weights[layer_index] + (1 - beta2) * (weight_gradients[layer_index] ** 2)
                    second_moment_biases[layer_index] = beta2 * second_moment_biases[layer_index] + (1 - beta2) * (bias_gradients[layer_index] ** 2)

                    # Compute bias-corrected moment estimates
                    first_moment_weights_hat[layer_index] = first_moment_weights[layer_index] / (1 - np.power(beta1, epoch + 1))
                    first_moment_biases_hat[layer_index] = first_moment_biases[layer_index] / (1 - np.power(beta1, epoch + 1))
                    second_moment_weights_hat[layer_index] = second_moment_weights[layer_index] / (1 - np.power(beta2, epoch + 1))
                    second_moment_biases_hat[layer_index] = second_moment_biases[layer_index] / (1 - np.power(beta2, epoch + 1))

                    # Compute NAdam update terms
                    weight_update_term = (beta1 * first_moment_weights_hat[layer_index] + (1 - beta1) * weight_gradients[layer_index] / (1 - np.power(beta1, epoch + 1)))
                    bias_update_term = (beta1 * first_moment_biases_hat[layer_index] + (1 - beta1) * bias_gradients[layer_index] / (1 - np.power(beta1, epoch + 1)))

                    # Update weights and biases using NAdam optimization
                    self.weights[layer_index] -= (learning_rate / np.sqrt(second_moment_weights_hat[layer_index] + epsilon)) * weight_update_term + weight_decay * self.weights[layer_index]
                    self.biases[layer_index] -= (learning_rate / np.sqrt(second_moment_biases_hat[layer_index] + epsilon)) * bias_update_term

            # Calculate and log loss at specific intervals
            if self.config["epochs"] == 10 or self.config["epochs"] == 5:
                self.calculate_loss(training_data, training_labels, validation_data, validation_labels, epoch)

    def gradient_descent(self,x_train_data, y_train_data, x_validation_data, y_validation_data):
        if(self.config["optimizer"] == "sgd"):
            self.stochastic_gradient_descent(x_train_data, y_train_data, x_validation_data, y_validation_data)
        elif(self.config["optimizer"] == "momentum"):
            self.momentum_based_gradient_descent(x_train_data, y_train_data, x_validation_data, y_validation_data)
        elif(self.config["optimizer"] == "nestrov"):
            self.nesterov_gradient_descent(x_train_data, y_train_data, x_validation_data, y_validation_data)
        elif(self.config["optimizer"] == "rmsprop"):
            self.rmsprop_optimization(x_train_data, y_train_data, x_validation_data, y_validation_data)
        elif(self.config["optimizer"] == "adam"):
            self.adam_optimization(x_train_data, y_train_data, x_validation_data, y_validation_data)
        elif(self.config["optimizer"] == "nadam"):
            self.nadam_optimization(x_train_data, y_train_data, x_validation_data, y_validation_data)

    # Method to calculate training and validation loss and accuracy
    def calculate_loss(self, train_data, train_labels, validation_data, validation_labels, epoch=0):
        train_correct = 0
        train_loss = 0
        validation_correct = 0
        validation_loss = 0
        epsilon = 1e-10  # Small constant to avoid log(0)

        # Calculate training loss and accuracy
        for i in range(len(train_data)):
            activations, _ = self.forward_propagate(train_data[i])
            predicted_class = np.argmax(activations[self.total_layers - 1])
            true_class = train_labels[i]

            if predicted_class == true_class:
                train_correct += 1

            if self.config["loss"] == "cross_entropy":
                log_value = max(activations[self.total_layers - 1][true_class, 0], epsilon)
                train_loss += -math.log10(log_value)
            elif self.config["loss"] == "mean_squared_error":
                one_hot_label = np.zeros((10, 1))
                one_hot_label[true_class] = 1
                train_loss += np.sum((activations[self.total_layers - 1] - one_hot_label) ** 2)

        # Calculate validation loss and accuracy
        for i in range(len(validation_data)):
            activations, _ = self.forward_propagate(validation_data[i])
            predicted_class = np.argmax(activations[self.total_layers - 1])
            true_class = validation_labels[i]

            if predicted_class == true_class:
                validation_correct += 1

            if self.config["loss"] == "cross_entropy":
                log_value = max(activations[self.total_layers - 1][true_class, 0], epsilon)
                validation_loss += -math.log10(log_value)
            elif self.config["loss"] == "mean_squared_error":
                one_hot_label = np.zeros((10, 1))
                one_hot_label[true_class] = 1
                validation_loss += np.sum((activations[self.total_layers - 1] - one_hot_label) ** 2)

        # Compute accuracy and average loss
        train_accuracy = train_correct / len(train_data)
        validation_accuracy = validation_correct / len(validation_data)
        train_loss /= len(train_data)
        validation_loss /= len(validation_data)

        # Print results
        print(f"Epoch: {epoch}, Train Accuracy: {train_accuracy}, Train Loss: {train_loss}, "
              f"Validation Accuracy: {validation_accuracy}, Validation Loss: {validation_loss}")

        # Log results to wandb if applicable
        if self.config["epochs"] == 10  or self.config["epochs"] == 5:
            wandb.log({
                "train_accuracy": train_accuracy,
                "train_loss": train_loss,
                "val_accuracy": validation_accuracy,
                "val_loss": validation_loss,
                "epoch": epoch
            })


    # Method to plot confusion matrix
    def generate_confusion_matrix(self, test_data, test_labels):
        predicted_labels = []
        for i in range(len(test_data)):
            activations, _ = self.forward_propagate(test_data[i])
            predicted_class = np.argmax(activations[self.total_layers - 1])
            predicted_labels.append(predicted_class)

        # Create confusion matrix
        confusion_matrix = np.zeros((10, 10))
        for i in range(len(test_labels)):
            confusion_matrix[test_labels[i]][predicted_labels[i]] += 1

        # Plot confusion matrix
        class_names = ['Ankle boot', 'T-shirt/top', 'Dress', 'Pullover', 'Sneaker', 'Sandal', 'Trouser', 'Shirt', 'Coat', 'Bag']
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, cmap=plt.cm.Blues, fmt='.2f', xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.savefig('confusion_matrix.png')
        wandb.log({"Confusion Matrix": wandb.Image('confusion_matrix.png')})


    # Method to update model parameters
    def update_model_parameters(self, weight_gradients, bias_gradients, learning_rate):
        for i in range(self.total_layers):
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network with specified hyperparameters.")
    parser.add_argument("-wp", "--wandb_project", type=str, default="CS24M027_Deep_Learning_Assignment_1", help="Project name for Weights & Biases")
    parser.add_argument("-we", "--wandb_entity", type=str, default="", help="Wandb Entity for tracking experiments")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, default="nadam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.8, help="Momentum for SGD with momentum or NAG")
    parser.add_argument("-beta", "--beta", type=float, default=0.99, help="Beta for RMSProp")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.8, help="Beta1 for Adam and Nadam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.99, help="Beta2 for Adam and Nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8, help="Epsilon for optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier", choices=["random", "xavier"], help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of neurons in hidden layers")
    parser.add_argument("-a", "--activation", type=str, default="relu", choices=["identity", "sigmoid", "tanh", "relu"], help="Activation function")
    return parser.parse_args()

import wandb

def main():
    arguments = parse_args()

    # Update hyperparameter defaults with command-line arguments
    hyperparameter_defaults = {
        "epochs": arguments.epochs,
        "hidden_layers": arguments.num_layers,
        "hl_size": arguments.hidden_size,
        "weight_decay": arguments.weight_decay,
        "learning_rate": arguments.learning_rate,
        "optimizer": arguments.optimizer,
        "batch_size": arguments.batch_size,
        "initialization": arguments.weight_init,
        "activation": arguments.activation,
        "loss": arguments.loss,
        "wandb_project": arguments.wandb_project,
        "wandb_entity": arguments.wandb_entity,
        "momentum_beta": arguments.momentum,
        "rms_beta": arguments.beta,
        "beta1": arguments.beta1,
        "beta2": arguments.beta2,
        "eps": arguments.epsilon
    }

    # Construct run name using key hyperparameters
    run_name = (
        f"opt_{hyperparameter_defaults['optimizer']}_"
        f"lr_{hyperparameter_defaults['learning_rate']}_"
        f"bs_{hyperparameter_defaults['batch_size']}_"
        f"ep_{hyperparameter_defaults['epochs']}_"
        f"nl_{hyperparameter_defaults['hidden_layers']}_"
        f"hs_{hyperparameter_defaults['hl_size']}_"
        f"act_{hyperparameter_defaults['activation']}_"
        f"loss_{hyperparameter_defaults['loss']}"
    )

    # Define sweep configuration
    sweep_config = {
        "name": "CS24M027",  # Default sweep name
        "method": "bayes",  # Search method: grid, random, or bayesian
        "metric": {"name": "val_accuracy", "goal": "maximize"},  
        "parameters": {
            "learning_rate": {"values": [0.001, 0.01, 0.1]},
            "batch_size": {"values": [32, 64, 128]},
            "optimizer": {"values": ["sgd", "adam", "nadam"]},
            "activation": {"values": ["relu", "tanh", "sigmoid"]},
            "epochs": {"values": [5, 10, 15]},
            "hidden_layers": {"values": [2, 4, 6]},
            "hl_size": {"values": [64, 128, 256]},
        },
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=hyperparameter_defaults["wandb_project"])

    # Function to run a single sweep iteration
    def train_sweep():
        # Initialize wandb run for the sweep
        wandb_run = wandb.init(
            project=hyperparameter_defaults["wandb_project"],
            entity=hyperparameter_defaults["wandb_entity"],
            config=hyperparameter_defaults,
            name=run_name
        )

        # Load dataset
        if arguments.dataset == "mnist":
            (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
        else:
            (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

        # Preprocess data
        x_total_train_processed = preprocess_image_data(train_data)
        x_test_processed = preprocess_image_data(test_data)

        # Split into training and validation sets
        x_train_final, x_validation_final, y_train_final, y_validation_final = train_test_split(
            x_total_train_processed, train_labels, train_size=0.9, random_state=42
        )

        # Initialize neural network
        neural_network = NeuralNetwork(784, 10, config=hyperparameter_defaults)

        # Train the network
        neural_network.gradient_descent(x_train_final, y_train_final, x_validation_final, y_validation_final)

        # Generate confusion matrix
        neural_network.generate_confusion_matrix(x_test_processed, test_labels)

    # Run the sweep
    wandb.agent(sweep_id, function=train_sweep)

if __name__ == "__main__":
    main()
