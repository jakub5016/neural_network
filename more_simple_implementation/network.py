from random import random
import numpy as np 
import pandas as pd
import json

class Network():
    def __init__(self, network_size):
        self.network_size = network_size

        self.neurons = []
        self.weights = []
        self.biases = []


    def mount(self):
        for index, item in enumerate(self.network_size):
            self.neurons.append(np.array([0] * item))

            if index > 0:
                self.weights.append(np.array([[random()]*item] * self.network_size[index-1]))
                self.biases.append(np.array([[random()]*item] * self.network_size[index-1]))
            else:
                self.weights.append(np.array([]))
                self.biases.append(np.array([]))

        for index, item in enumerate(self.neurons):
            if index > 0:
                item = np.matmul(self.neurons[index-1], self.weights[index])

        print("NEURONS:", self.neurons)
        print("WEIGHTS: ", self.weights)
        print("BIASES:", self.biases)


    def save(self):
        weights_as_lists = [weight.tolist() for weight in self.weights]

        with open('weights.json', 'w') as f:
            json.dump(weights_as_lists, f)

    def evaluate(self, input):

        if len(input) == len(self.neurons[0]):
            self.neurons[0] = input 

        else:
            print("Wrong input size")
            return -1
        

        for layer_index in range(len(self.neurons)):
            if layer_index > 0:
                self.neurons[layer_index] = np.matmul(self.neurons[layer_index-1], self.weights[layer_index])
                print(f"Layer number: {layer_index}, {self.neurons[layer_index]}, after matmul")

                for elem_index in range(len(self.neurons[layer_index])):
                    for i in self.biases[layer_index]:
                        self.neurons[layer_index][elem_index] += i[elem_index]
                        self.neurons[layer_index][elem_index] = np.tanh(self.neurons[layer_index][elem_index])
                
                print(f"Layer number: {layer_index}, {self.neurons[layer_index]}, after bias and tanh")

        return self.neurons[-1]

if __name__ == "__main__":
    network = Network(network_size=[2,2,2])
    network.mount()
    print(network.evaluate([1,1]))