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
        
        for index, layer in enumerate(self.neurons):
            if index > 0:
                layer = np.matmul(self.neurons[index-1], self.weights[index])

                print(layer)
                
                for elem_index, element in enumerate(layer):
                    for bias_list in self.biases[index]:
                        element = element + bias_list[elem_index]


                    element = np.tanh(element)



if __name__ == "__main__":
    network = Network(network_size=[2,2,3])
    network.mount()
    network.evaluate([2,2])