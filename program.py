import NeuralNetwork as nn
import numpy as np

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

# (input layer, 2 hidden layers, output layer )
layer_sizes = (784, 15, 15, 10)

# number of images used for training, the rest is used for testing the performance
training_set_size = 5000

training_set_images = training_images[:training_set_size]
training_set_labels = training_labels[:training_set_size]

test_set_images = training_images[training_set_size:]
test_set_labels = training_labels[training_set_size:]

# initialize
net = nn.NeuralNetwork(layer_sizes)


# eval performance without training
net.print_accuracy(test_set_images, test_set_labels)
net.calculate_average_cost(test_set_images, test_set_labels)

net.train_sgd(training_set_images, training_set_labels, 2, 10, 4.0)


# eval performance with 1st training
net.print_accuracy(test_set_images, test_set_labels)
net.calculate_average_cost(test_set_images, test_set_labels)

net.train_sgd(training_set_images, training_set_labels, 8, 20, 2.0)

# eval performance with 2nd training
net.print_accuracy(test_set_images, test_set_labels)
net.calculate_average_cost(test_set_images, test_set_labels)