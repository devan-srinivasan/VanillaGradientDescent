"""Main Python Script"""
from Network_optimized import Network
import numpy as np
from time import time


def load_sklearn() -> np.array:
    """load sklearn digits"""
    from sklearn.datasets import load_digits
    d = load_digits()
    inputs = [np.array(img).reshape(64, 1) for img in d.data]
    targets = np.array([np.array([[int(i == t)] for i in range(10)]).reshape(10, 1)
                        for t in d.target])
    return list(zip(inputs, targets))


def test_digits(network: Network, test_pairs: np.array) -> np.array:
    """test the network using sklearn data"""
    bitmap = [0] * 10
    ideal = [0] * 10
    for pair in test_pairs:
        network.backpropagation(pair[0], pair[1])
        o = network.feedforward(pair[0])
        a = o.argmax()
        b = pair[1].argmax()
        if a == b:
            bitmap[b] += 1
        ideal[b] += 1
    return bitmap, ideal


if __name__ == '__main__':
    # initialization
    net = Network([64, 30, 10])

    # Training
    training_amount = 1600  # out of 1797 total samples
    epochs = 10
    minibatch_size = 20
    learning_rate = 0.5

    start_time = time()
    data = load_sklearn()
    np.random.shuffle(data)
    testing_data = data[training_amount:]
    training_data = data[:training_amount]
    net.minibatch_gradient_descent(training_data, epochs, minibatch_size, learning_rate)

    end_time = time()
    minutes = int((end_time - start_time) // 60)
    seconds = end_time - start_time - minutes * 60
    print(f"==> training finished: {minutes}:{seconds} mins")

    # Testing
    start_time = time()
    num_testing = len(testing_data)
    test_result, ideal_result = test_digits(net, testing_data)
    end_time = time()
    minutes = int((end_time - start_time) // 60)
    seconds = end_time - start_time - minutes * 60
    print(f"==> testing finished: {minutes}:{seconds} mins")
    print(test_result)
    print(ideal_result)
    print(f"{sum(test_result) * 100 / num_testing} %")
