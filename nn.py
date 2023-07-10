import numpy as np
import scipy.special # sigmoid function

from typing import List


class NeuralNetMH:
    """
    A neural network with multiple hidden layers.
    """
    def __init__(self, input_nodes: int, hidden_nodes: List[int], output_nodes: int, learning_rate: float):
        """
        Initialize the neural network.

        :param input_nodes: The number of input nodes.
        :param hidden_nodes: A list of the number of hidden nodes in each hidden layer.
        :param output_nodes: The number of output nodes.
        :param learning_rate: The learning rate of the neural network.
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate
        self.activation_function = lambda x: scipy.special.expit(x)

        self.weights = []
        for i in range(len(hidden_nodes)):
            if i == 0:
                self.weights.append(np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes[i], self.input_nodes)))
            else:
                self.weights.append(np.random.normal(0.0, pow(self.hidden_nodes[i - 1], -0.5), (self.hidden_nodes[i], self.hidden_nodes[i - 1])))
        self.weights.append(np.random.rand(self.output_nodes, self.hidden_nodes[-1]) - 0.5)

    
    def train(self, inputs: List[float], targets: List[float]):
        """
        Train the neural network with the given inputs and targets.
        
        :param inputs: The inputs to the neural network.
        :param targets: The target outputs of the neural network.
        """
        targets = np.array(targets, ndmin=2).T
        hidden_outputs = []
        for i in range(len(self.hidden_nodes)):
            if i == 0:
                hidden_outputs.append(self.activation_function(np.dot(self.weights[i], np.array(inputs, ndmin=2).T)))
            else:
                hidden_outputs.append(self.activation_function(np.dot(self.weights[i], hidden_outputs[i - 1])))
        final_outputs = self.query(inputs)

        output_errors = targets - final_outputs
        hidden_errors = []
        for i in range(len(self.hidden_nodes) - 1, -1, -1):
            if i == len(self.hidden_nodes) - 1:
                hidden_errors.append(np.dot(self.weights[i].T, output_errors))
            else:
                hidden_errors.append(np.dot(self.weights[i].T, hidden_errors[-1]))

        for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1:
                self.weights[i] += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs[i]))
            else:
                self.weights[i] += self.learning_rate * np.dot((hidden_errors[len(self.hidden_nodes) - i - 2] * hidden_outputs[i] * (1.0 - hidden_outputs[i])), np.transpose(inputs))


    def query(self, inputs: List[float]) -> List[float]:
        """
        Query the neural network with the given inputs.

        :param inputs: The inputs to the neural network.
        :return: The outputs of the neural network.
        """
        inputs = np.array(inputs, ndmin=2).T

        hidden_inputs = []
        hidden_outputs = []
        for i in range(len(self.hidden_nodes)):
            if i == 0:
                hidden_inputs.append(np.dot(self.weights[i], inputs))
            else:
                hidden_inputs.append(np.dot(self.weights[i], hidden_outputs[i - 1]))
            hidden_outputs.append(self.activation_function(hidden_inputs[i]))

        final_inputs = np.dot(self.weights[-1], hidden_outputs[-1])
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
