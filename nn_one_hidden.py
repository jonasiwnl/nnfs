import numpy as np
import scipy.special # sigmoid function
from typing import List


class NeuralNetOH:
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
        self.inverse_activation_function = lambda x: scipy.special.logit(x) # inverse sigmoid function

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


    def backquery(self, outputs: List[float]) -> List[float]:
        """
        Backquery the neural network with the given outputs.
        
        :param outputs: The outputs of the neural network.
        :return: The inputs of the neural network.
        """
        outputs = np.array(outputs, ndmin=2).T

        final_inputs = self.inverse_activation_function(outputs)
        hidden_outputs = np.dot(self.weights_ho.T, final_inputs)

        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        inputs = np.dot(self.weights_ih.T, hidden_inputs)

        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs
    

    def save(self, filename: str):
        """
        Save the neural network to a file.

        :param filename: The filename to save the neural network to.
        """
        np.savez(filename, self.input_nodes, self.hidden_nodes, self.output_nodes, self.learning_rate, self.weights_ih, self.weights_ho)

    
    def load(self, filename: str):
        """
        Load the neural network from a file.

        :param filename: The filename to load the neural network from.
        """
        data = np.load(filename, allow_pickle=True)
        self.input_nodes = data["arr_0"]
        self.hidden_nodes = data["arr_1"]
        self.output_nodes = data["arr_2"]
        self.learning_rate = data["arr_3"]
        self.weights_ih = data["arr_4"]
        self.weights_ho = data["arr_5"]
