from typing import List

from nn_one_hidden import NeuralNetOH


def epoch_single_train(n: NeuralNetOH, training_data: List[str], epochs: int):
    for i in range(epochs):
        n.train(training_data[i][1:], training_data[i][0])


def main():
    input_nodes = 5
    hidden_nodes = 100
    output_nodes = 1
    learning_rate = 0.3

    n = NeuralNetOH(input_nodes, hidden_nodes, output_nodes, learning_rate)

    """
    training_data = None
    with open("mnist/mnist_train_100.csv", "r") as f:
        training_data = f.readlines()
    """

    epoch_single_train(n, training_data, 100)

    output = n.query([1.0, 0.5, -1.5, 0.8, 0.5])
    print(output)
    print(n.backquery(output))


if __name__ == "__main__":
    main()
