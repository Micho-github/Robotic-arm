import math
import random
import pickle
from utils.helpers import relu, relu_derivative, linear, linear_derivative, tanh_func, tanh_derivative

class Node:
    def __init__(self, activation_func=relu, activation_derivative=relu_derivative):
        self.weights = []
        self.bias = random.uniform(-0.5, 0.5)
        self.output = 0
        self.delta = 0
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative
        self.z = 0

    def initialize_weights(self, num_inputs):
        # Xavier initialization
        limit = math.sqrt(6.0 / num_inputs)
        self.weights = [random.uniform(-limit, limit) for _ in range(num_inputs)]

    def forward(self, inputs):
        self.z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.output = self.activation_func(self.z)
        return self.output

class Layer:
    def __init__(self, num_nodes, activation_func=relu, activation_derivative=relu_derivative):
        self.nodes = [Node(activation_func, activation_derivative) for _ in range(num_nodes)]
        self.outputs = []

    def initialize_weights(self, num_inputs):
        for node in self.nodes:
            node.initialize_weights(num_inputs)

    def forward(self, inputs):
        self.outputs = [node.forward(inputs) for node in self.nodes]
        return self.outputs

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.learning_rate = 0.001
        self.architecture = []  # Store architecture info for reconstruction

    def add_layer(self, num_nodes, activation_func=relu, activation_derivative=relu_derivative):
        layer = Layer(num_nodes, activation_func, activation_derivative)
        self.layers.append(layer)

        # Store architecture info
        func_name = activation_func.__name__ if hasattr(activation_func, '__name__') else 'unknown'
        self.architecture.append({
            'num_nodes': num_nodes,
            'activation': func_name
        })

    def initialize_network(self, input_size):
        prev_size = input_size
        for layer in self.layers:
            layer.initialize_weights(prev_size)
            prev_size = len(layer.nodes)

    def forward_propagation(self, inputs):
        current_inputs = inputs
        for layer in self.layers:
            current_inputs = layer.forward(current_inputs)
        return current_inputs

    def backward_propagation(self, inputs, targets):
        # Calculate output layer deltas
        output_layer = self.layers[-1]
        for i, node in enumerate(output_layer.nodes):
            error = targets[i] - node.output
            node.delta = error * node.activation_derivative(node.z)

        # Calculate hidden layer deltas
        for layer_idx in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]

            for i, node in enumerate(current_layer.nodes):
                error = sum(next_node.weights[i] * next_node.delta
                           for next_node in next_layer.nodes)
                node.delta = error * node.activation_derivative(node.z)

        # Update weights and biases
        current_inputs = inputs
        for layer in self.layers:
            for node in layer.nodes:
                for j in range(len(node.weights)):
                    node.weights[j] += self.learning_rate * node.delta * current_inputs[j]
                node.bias += self.learning_rate * node.delta
            current_inputs = layer.outputs

    def train(self, training_data, epochs=100):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, targets in training_data:
                outputs = self.forward_propagation(inputs)
                loss = sum((target - output) ** 2 for target, output in zip(targets, outputs)) / len(targets)
                epoch_loss += loss
                self.backward_propagation(inputs, targets)

            avg_loss = epoch_loss / len(training_data)
            losses.append(avg_loss)

            if epoch % 20 == 0:
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

        return losses

    def predict(self, inputs):
        return self.forward_propagation(inputs)

    def save_to_file(self, filename):
        """Save the neural network to a file"""
        save_data = {
            'architecture': self.architecture,
            'learning_rate': self.learning_rate,
            'weights_and_biases': []
        }

        # Extract weights and biases from each layer
        for layer in self.layers:
            layer_data = []
            for node in layer.nodes:
                layer_data.append({
                    'weights': node.weights.copy(),
                    'bias': node.bias
                })
            save_data['weights_and_biases'].append(layer_data)

        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load_from_file(cls, filename):
        """Load a neural network from a file"""
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)

        # Create new network
        nn = cls()
        nn.learning_rate = save_data['learning_rate']

        # Reconstruct architecture
        activation_map = {
            'relu': (relu, relu_derivative),
            'linear': (linear, linear_derivative),
            'tanh_func': (tanh_func, tanh_derivative)
        }

        for layer_info in save_data['architecture']:
            activation_name = layer_info['activation']
            if activation_name in activation_map:
                activation_func, activation_derivative = activation_map[activation_name]
            else:
                activation_func, activation_derivative = relu, relu_derivative

            nn.add_layer(layer_info['num_nodes'], activation_func, activation_derivative)

        # Initialize network structure
        nn.initialize_network(input_size=2)

        # Load weights and biases
        for layer_idx, layer_data in enumerate(save_data['weights_and_biases']):
            for node_idx, node_data in enumerate(layer_data):
                nn.layers[layer_idx].nodes[node_idx].weights = node_data['weights']
                nn.layers[layer_idx].nodes[node_idx].bias = node_data['bias']

        return nn