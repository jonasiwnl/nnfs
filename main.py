from nn_one_hidden import NeuralNetOH


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

    print(n.query([1.0, 0.5, -1.5, 0.8, 0.5]))


if __name__ == "__main__":
    main()
