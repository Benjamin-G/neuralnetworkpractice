import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s) / s[1] ** .5 for s in weight_shapes]
        self.biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]

        self.num_layers = len(layer_sizes)
        self.activations = np.asarray([np.zeros(size) for size in layer_sizes])

    def feedforward(self, sample):
        # Feeding a sample into the network

        # Input layer
        self.activations[0] = sample

        for i in range(self.num_layers - 1):
            z = np.matmul(self.weights[i], self.activations[i]) + self.biases[i]
            self.activations[i + 1] = self.activation_function(z)

        # Return Output layer
        return self.activations[-1]

    def training_sgd(self, training_images, training_labels, epochs, batch_size, learning_rate):
        # Train using Stochastic Gradient Decent

        training_data = [(x, y) for x, y in zip(training_images, training_labels)]

        print("start training")
        for j in range(epochs):
            batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]

            for batch in batches:
                self.update_batch(batch, learning_rate)

        print("epoch {0} complete".format(j))

    def update_batch(self, batch, learning_rate):
        # direction of gradient determines the change to weights and biases
        bias_gradients = [np.zeros(b.shape) for b in self.biases]
        weights_gradients = [np.zeros(w.shape) for w in self.weights]

        for sample, label in batch:
            # calc the direction of the gradient
            bias_deltas, weight_deltas = self.back_propagation(sample, label)

            # combine
            bias_gradients = [b_gradient + b_delta for b_gradient, b_delta in zip(bias_gradients, bias_deltas)]
            weight_gradients = [w_gradient + w_delta for w_gradient, w_delta in zip(weight_gradients, weight_deltas)]

        self.biases = [b - (learning_rate / len(batch)) * b_gradient for b, b_gradient in
                       zip(self.biases, bias_gradients)]
        self.weights = [w - (learning_rate / len(batch)) * w_gradient for w, w_gradient in
                        zip(self.weights, weight_gradients)]

    def back_propagation(self, sample, label):
        # sample -> network and calc changes so that the cost is minimized

        bias_deltas = [np.zeros(b.shape) for b in self.biases]
        weight_deltas = [np.zeros(w.shape) for w in self.weights]

        self.feedforward(sample)

        # theory by 3Blue1Brown: https://youtu.be/tIeHLnjs5U8
        L = -1

        partial_deltas = self.cost_function_derivative(self.activations[L], label) * self.cost_function_derivative(
            self.activations[L])

        bias_deltas[L] = partial_deltas
        weight_deltas[L] = np.dot(partial_deltas, self.activations[L - 1].T)

        while L > -self.num_layers + 1:
            previous_layer_deltas = np.dot(self.weights[L].T, partial_deltas)

            partial_deltas = previous_layer_deltas * self.activation_function_derivative(self.activations[L - 1])
            bias_deltas[L - 1] = partial_deltas
            weight_deltas[L - 1] = np.dot(partial_deltas, self.activations[L - 2].T)

            L -= 1

        return bias_deltas, weight_deltas

    def calculate_average_cost(self, samples, labels):
        predictions = self.feedforward(samples)
        average_cost = sum([self.cost_function(p, l) for p, l in zip(predictions, labels)]) / len(samples)
        print("average cost: {0}".format(average_cost))

    '''
        legacy predict 
        def predict(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation(np.matmul(w, a) + b)
        return a
    '''

    def print_accuracy(self, samples, labels):
        predictions = [self.feedforward(sample) for sample in samples]
        num_correct = sum([np.argmax(p) == np.argmax(l) for p, l in zip(predictions, labels)])
        print('{0}/{1} accuracy: {2}%'.format(num_correct, len(samples), (num_correct / len(samples) * 100)))

    '''
        Activation function + derivative (can pick different functions like ReLu
    '''

    @staticmethod
    def activation_function(z):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        return sigmoid(z)

    @staticmethod
    def activation_function_derivative(a):
        def dsigmoid(x):
            return x * (1 - x)

        return dsigmoid(a)

    '''
        Cost function + derivative
    '''

    @staticmethod
    def cost_function(output, y):
        def sum_of_squares(out, x):
            return sum((a - b) ** 2 for a, b in zip(out, x))[0]

        return sum_of_squares(output, y)

    @staticmethod
    def cost_function_derivative(output, y):
        def sum_of_squares_prime(out, x):
            return 2 * (out - x)

        return sum_of_squares_prime(output, y)