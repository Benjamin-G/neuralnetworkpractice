import NeuralNetwork as nn
import numpy as np

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

layer_sizes = (784, 5, 10)


net = nn.NeuralNetwork(layer_sizes)
net.print_accuracy(training_images, training_labels)




