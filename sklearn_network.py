"""Main Python Script"""
from Network import Network
import numpy as np
from time import time


def load_sklearn() -> np.array:
    """load sklearn digits"""
    from sklearn.datasets import load_digits
    d = load_digits()
    targets = np.array([[[int(i == t)] for i in range(10)] for t in d.target])
    return list(zip(d.data, targets))


def test_digits(network: Network, test_pairs: np.array) -> np.array:
    """test the network using sklearn data"""
    bitmap = [0] * 10
    ideal = [0] * 10
    for pair in test_pairs:
        network.set_input(pair[0])
        network.feedforward()
        o = network.get_output()
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
    start_time = time()
    data = load_sklearn()
    np.random.shuffle(data)
    testing_data = data[1600:]
    training_data = data[:1600]
    net.minibatch_gradient_descent(training_data, 40, 20, 0.5)

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
