"""Neural Network"""
import numpy as np

class Network:
    """This network doesn't have a Layer class as my other implementation did
    This approach is better for implementing mini-batch gradient descent"""

    weights: np.array
    bias: np.array
    activation: np.array
    struct: np.array

    def __init__(self, structure: list) -> None:
        """initialize the network, structure is a list of integers
        here we assume structure[0] is the size of input and structure[-1] is size of output"""
        assert (len(structure) >= 2)
        self.struct = np.array(structure)
        w, b, a = [], [], []
        for i in range(len(structure)):
            if i > 0:
                w.append(np.random.randn(self.struct[i], self.struct[i - 1]) \
                               * np.sqrt(2 / self.struct[i - 1]))
                b.append(np.zeros((self.struct[i], 1)))
                a.append(np.zeros((self.struct[i], 1)))
            else:
                w.append(np.zeros((self.struct[i], 1)))
                b.append(np.zeros((self.struct[i], 1)))
                a.append(np.zeros((self.struct[i], 1)))
        self.weights = w
        self.bias = b
        self.activation = a

    def set_input(self, input_data: np.array) -> None:
        """set the input layer"""
        assert len(input_data) == len(self.activation[0])
        self.activation[0] = np.array([[e] for e in input_data])

    def get_output(self) -> np.array:
        """return the output of the network"""
        return self.activation[-1]

    def feedforward(self) -> None:
        """feed forward input to output"""
        # set the first layer based on input layer
        for l in range(1, len(self.struct)):
            self.activation[l] = sig(np.matmul(self.weights[l], self.activation[l - 1])
                                            + self.bias[l])

    def backpropagation(self, exp: np.array) -> tuple[np.array, np.array]:
        """backpropagation algorithm
        passes backwards through the network updating the error values
        return gradient with respect to weights and with respect to biases

        we use the CURRENT output of the network in comparison to the exp array
        """
        assert (len(exp) == len(self.activation[-1]))

        # 1) forward, calculate z values and error (delta) values for last layer
        error = [np.zeros((self.struct[l], 1)) for l in range(len(self.struct))]
        z_values = [
            np.matmul(self.weights[l], self.activation[l - 1])
            + self.bias[l] for l in range(1, len(self.struct))
        ]
        z_values.insert(0, np.zeros((self.struct[0], 1)))

        error[-1] = np.multiply(
            self.activation[-1] - exp,
            sig_prime(z_values[-1])
        )

        # 2) backwards, calculate error (delta) values for previous layers
        for l in range(len(self.struct) - 2, 0, -1):
            error[l] = np.multiply(
                np.matmul(
                    np.transpose(self.weights[l + 1]),
                    error[l + 1]
                ),
                sig_prime(z_values[l])
            )

        # 3) calculate partial derivatives
        gradient_w = [np.zeros((self.struct[l], self.struct[l - 1]))
                      for l in range(1, len(self.struct))]
        gradient_w.insert(0, np.zeros((self.struct[0], 1)))

        for l in range(1, len(self.struct)):
            num_prev, num_next = self.struct[l - 1], self.struct[l]

            # this is the gradient matrices for the weights
            for i in range(num_next):
                for j in range(num_prev):
                    gradient_w[l][i][j] = error[l][i] * self.activation[l - 1][j]

            # gradient vector for weights and biases is returned
            gradient_b = error
            return (gradient_w, gradient_b)

    def update_single(self, w_change: np.array, b_change: np.array, lr: float) -> None:
        """stochastic gradient descent"""
        for i in range(len(self.struct)):
            self.weights[i] -= lr * w_change[i]
            self.bias[i] -= lr * b_change[i]

    def update_minibatch(self, minibatch: np.array, lr: float) -> None:
        """mini-batch gradient descent"""
        weight_gradients = [np.zeros(w.shape) for w in self.weights]
        bias_gradients = [np.zeros(b.shape) for b in self.bias]
        for n_in, expected in minibatch:
            self.set_input(n_in)
            self.feedforward()
            nw, nb = self.backpropagation(expected)
            for n_w, w_g in zip(nw, weight_gradients):
                w_g += n_w
            for n_b, b_g in zip(nb, bias_gradients):
                b_g += n_b

        # now that accumulated gradient is calculated, we can take the average x learning_rate
        for w, w_g in zip(self.weights, weight_gradients):
            w -= ((lr / len(minibatch)) * w_g)
        for b, b_g in zip(self.bias, bias_gradients):
            b -= ((lr / len(minibatch)) * b_g)

    def minibatch_gradient_descent(self, training_data, epochs, mini_batch_size, lr) -> None:
        """Minibatch gradient descent"""
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_minibatch(mini_batch, lr)


def sig(x) -> float:
    """Sigmoid of x"""
    return 1 / (1 + np.exp(-x))


def sig_prime(y: float) -> float:
    """derivative of the sigmoid function"""
    return sig(y) * (1 - sig(y))


def square_cost(out: np.array, exp: np.array) -> np.array:
    """calculate squared cost of output"""
    diff = (out - exp) ** 2
    return sum(e for e in diff)
