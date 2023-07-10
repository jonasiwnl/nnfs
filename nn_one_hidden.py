import numpy as np
import scipy.special # sigmoid function
from typing import List


class NeuralNet:
    """
    A neural network with one hidden layer.
    """
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int, learning_rate: float):
        """
        Initialize the neural network.

        :param input_nodes: The number of input nodes.
        :param hidden_nodes: The number of hidden nodes.
        :param output_nodes: The number of output nodes.
        :param learning_rate: The learning rate of the neural network.
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate
        self.activation_function = lambda x: scipy.special.expit(x) # sigmoid function

        self.weights_ih = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5


    def train(self, inputs, targets):
        """
        Train the neural network with the given inputs and targets.

        :param inputs: The inputs to the neural network.
        :param targets: The target outputs of the neural network.
        """
        targets = np.array(targets, ndmin=2).T
        hidden_outputs = self.activation_function(np.dot(self.weights_ih, np.array(inputs, ndmin=2).T))
        final_outputs = self.query(inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_ho.T, output_errors)

        self.weights_ho += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.weights_ih += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))


    def query(self, inputs: List[float]) -> List[float]:
        """
        Query the neural network with the given inputs.

        :param inputs: The inputs to the neural network.
        :return: The outputs of the neural network.
        """
        inputs = np.array(inputs, ndmin=2).T

        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
