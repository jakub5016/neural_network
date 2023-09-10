from random import random
import numpy as np 
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
                self.biases.append(np.array([1] * item))
            else:
                self.weights.append(np.array([]))
                self.biases.append(np.array([]))

        for index, item in enumerate(self.neurons):
            if index > 0:
                item = np.matmul(self.neurons[index-1], self.weights[index])

        self.print_network_status()


    def print_network_status(self):
        print("\nNEURONS:", self.neurons)
        print("WEIGHTS: ", self.weights)
        print("BIASES:", self.biases, "\n")

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
                for elem_index in range(len(self.neurons[layer_index])):
                    self.neurons[layer_index][elem_index] += self.biases[layer_index][elem_index]
                    
                    # Use softmax on last index
                    # if layer_index ==(len(self.neurons) -1):
                    #     self.neurons[layer_index][elem_index] = np.exp(self.neurons[layer_index][elem_index])/ np.sum(np.exp(self.neurons[layer_index][elem_index]), dtype=np.float64)
                    # else:
                    self.neurons[layer_index][elem_index] = np.tanh(self.neurons[layer_index][elem_index])
                
        return self.neurons[-1]
    
    def back_propagation(self, answer, learing_rate =0.1):
        if len(answer) == len(self.neurons[-1]):
            cost = np.square(answer - self.neurons[-1])
            print(f"Cost: {cost} \n")

            for index_L in range(len(self.neurons[-1])):
                bias_gradient = 0
                for index_L_1 in range(len(self.neurons[-2])):
                    
                    weight_gradient = 2*self.neurons[-2][index_L_1] * np.tan(self.neurons[-1][index_L]) * (self.neurons[-1][index_L] - answer[index_L])
                    # print(f"Weight gradient for neurons: {index_L} in last layer, {index_L_1} in pervous layer ::: {weight_gradient}")
                    # print(f"Current weight: {self.weights[-1][index_L_1][index_L]}")
                    self.weights[-1][index_L_1][index_L] -= learing_rate * weight_gradient

                    bias_gradient += weight_gradient/self.neurons[-2][index_L_1] 
                    # print(f"Bias gradient ::: {bias_gradient}, devided by: {self.neurons[-2][index_L_1] }")

                self.biases[-1][index_L] -= learing_rate * bias_gradient

if __name__ == "__main__":
    network = Network(network_size=[2,2])
    network.mount()
    print(f"For [1,1] input: {network.evaluate([3,4])} \n")
    network.back_propagation(np.array([1, 0]))
    network.print_network_status()