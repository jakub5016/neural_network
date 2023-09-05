from random import random
import numpy as np 
network_size = [2,3,3]  

neurons = []
weights = []
biases = []

for index, item in enumerate(network_size):
    neurons.append(np.array([0] * item))

    if index > 0:
        weights.append(np.array([[random()]*item, [random()] * item]))
        biases.append(np.array([[random()]*item, [random()] * item]))
    else:
        weights.append([])
        biases.append([])


neurons[0] = np.array([1,1])

# What we want to multiply 
print(neurons[0])
print(weights[1])
print(np.matmul(neurons[0], weights[1]))

