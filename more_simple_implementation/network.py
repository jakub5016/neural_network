from random import random
import numpy as np 
import json
import matplotlib.pyplot as plt


class Network():
    def __init__(self, network_size):
        self.network_size = network_size

        self.neurons = []
        self.weights = []
        self.biases = []

        self.cost_polt = []


    def mount(self):
        for index, item in enumerate(self.network_size):
            self.neurons.append(np.array([0] * item))

            if index > 0:
                self.weights.append(np.array([[random()]*item] * self.network_size[index-1]))
                self.biases.append(np.array([1] * item, dtype=np.float64))
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
            self.cost_polt.append(np.sum(cost))

            for current_layer in range(len(self.network_size) -1):
                for index_L in range(len(self.neurons[-1 - current_layer])):
                    bias_gradient = 0
                    for index_L_1 in range(len(self.neurons[-2 - current_layer])):
                        
                        weight_gradient = 2*self.neurons[-2 - current_layer][index_L_1] * np.tan(self.neurons[-1 - current_layer][index_L]) * (self.neurons[-1][index_L] - answer[index_L])
                        # print(f"Weight gradient for neurons: {index_L} in last layer, {index_L_1} in pervous layer ::: {weight_gradient}")
                        # print(f"Current weight: {self.weights[-1][index_L_1][index_L]}")
                        self.weights[-1 - current_layer][index_L_1][index_L] -= learing_rate * weight_gradient

                        bias_gradient += weight_gradient/self.neurons[-2 - current_layer][index_L_1] 
                        # print(f"Bias gradient ::: {bias_gradient}, devided by: {self.neurons[-2][index_L_1] }")

                    self.biases[-1 - current_layer][index_L] -= float(learing_rate * bias_gradient)
    
    def train(self, in_out, learning_rate=0.1, n_eval =1):
        # in_out should be a table with inputs and corresponding outputs
        for i in range(n_eval):
            for example in in_out:
                print("Evaluating example:",example)
                self.evaluate(example[0])
                self.back_propagation(example[1], learning_rate)
            np.random.shuffle(in_out)
    
    def plot_cost(self):
        fig, ax = plt.subplots()

        ax.plot(self.cost_polt, linewidth=2.0)

        plt.show()

if __name__ == "__main__":
    # network = Network(network_size=[2, 2, 2])
    # network.mount()
    # print(f"For input: {network.evaluate([3,4])} \n")
    # network.back_propagation(np.array([1, 0]))
    # network.print_network_status()
    
    network = Network(network_size=[2, 1])
    network.mount()
    # AND expample
    network.train([
        [[1, 1], [1]], 
        [[1, 0.1], [0]],
        [[0.1, 1], [0]],
        [[0.1, 0.1], [0]]
        ], n_eval = 2000, learning_rate = 0.001)
    
    print(network.evaluate([1,1]))
    print(network.evaluate([1,0.1]))

    network.plot_cost()