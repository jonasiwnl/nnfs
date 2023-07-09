from nn import NeuralNet

def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3

    n = NeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)
    print(n.query([1.0, 0.5, -1.5]))

if __name__ == "__main__":
    main()
