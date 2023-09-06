from random import random
import numpy as np 
network_size = [2,1,5]  

neurons = []
weights = []
biases = []

for index, item in enumerate(network_size):
    neurons.append(np.array([0] * item))

    if index > 0:
        weights.append(np.array([[random()]*item] * network_size[index-1]))
        biases.append(np.array([[random()]*item] * network_size[index-1]))
    else:
        weights.append([])
        biases.append([])

print("NEURONS:", neurons, "\n")
print("WEIGHTS:", weights, "\n")

# What we want to multiply 

for index, item in enumerate(neurons):
    if index > 0:
        item = np.matmul(neurons[index-1], weights[index])

# print(np.matmul(np.array([0, 0, 0, 0]), weights[1]))