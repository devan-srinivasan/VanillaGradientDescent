"""Optimized network drawing on strategies online"""
import numpy as np


class Network:
    """Better network (hopefully faster)"""

    def __init__(self, structure) -> None:
        """initialize the network"""
        self.num_layers = len(structure)
        self.sizes = structure
        self.biases = [np.random.randn(i, 1) for i in structure[1:]]
        self.weights = [np.random.randn(structure[i+1], structure[i])
                        * np.sqrt(2 / structure[i])
                        for i in range(self.num_layers - 1)]

    def backpropagation(self, x, y) -> tuple:
        """backpropagation, returns the gradient vector

        first feedforward to calculate all the z value and activiation values
        then go backwards and calculate the partial derivatives"""
        delta_bias = [np.zeros(b.shape) for b in self.biases]
        errors = [np.zeros(b.shape) for b in self.biases]
        delta_weight = [np.zeros(w.shape) for w in self.weights]
        activations = [x]
        z_values = []

        # forward pass
        for i in range(self.num_layers - 1):
            z_values.append(np.matmul(self.weights[i], activations[i]) + self.biases[i])
            activations.append(sig(z_values[i]))

        # backward pass
        errors[-1] = np.multiply(self.cost_derivative(activations[-1], y), sig_prime(z_values[-1]))
        delta_bias[-1] = errors[-1]
        delta_weight[-1] = np.dot(errors[-1], np.transpose(activations[-2]))

        for l in range(self.num_layers - 2, 0, -1):
            errors[l-1] = np.multiply(np.matmul(np.transpose(self.weights[l]), errors[l]),
                                    sig_prime(z_values[l-1]))
            delta_bias[l-1] = errors[l-1]
            delta_weight[l-1] = np.matmul(errors[l-1], np.transpose(activations[l - 1]))
        return (delta_weight, delta_bias)


    def update_minibatch(self, minibatch: np.array, lr: float) -> None:
        """update weights and bias based on mini-batch gradient"""
        weight_gradients = [np.zeros(w.shape) for w in self.weights]
        bias_gradients = [np.zeros(b.shape) for b in self.biases]
        for x, y in minibatch:
            nw, nb = self.backpropagation(x, y)
            for n_w, w_g in zip(nw, weight_gradients):
                w_g += n_w
            for n_b, b_g in zip(nb, bias_gradients):
                b_g += n_b

        # now that accumulated gradient is calculated, we can take the average x learning_rate
        for w, w_g in zip(self.weights, weight_gradients):
            w -= ((lr / len(minibatch)) * w_g)
        for b, b_g in zip(self.biases, bias_gradients):
            b -= ((lr / len(minibatch)) * b_g)

    def cost_derivative(self, actual_out, expected_out):
        """derivative of cost function [(y' - y)^2]/2"""
        return actual_out - expected_out

    def feedforward(self, input_vector) -> np.array:
        """feedforward input and return output"""
        out_vector = input_vector
        for b, w in zip(self.biases, self.weights):
            out_vector = sig(np.dot(w, out_vector) + b)
        return out_vector

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
