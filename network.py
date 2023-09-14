from random import random
import numpy as np 
import json
import matplotlib.pyplot as plt
import time


class Network():
    def __init__(self, network_size, output_act_func="tanh"):
        self.network_size = network_size

        self.neurons = []
        self.weights = []
        self.biases = []

        self.cost_polt = []

        self.output_act_func = output_act_func

    def mount(self):
        """
        Sets random weights and biases for whole neural network.
        """
        for index, item in enumerate(self.network_size):
            self.neurons.append(np.array([0] * item))

            if index > 0:
                self.weights.append(np.random.normal(size=(self.network_size[index-1], item)))
                self.biases.append(np.array([0] * item, dtype=np.float64))
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
        """
        Saves weights in JSON file
        """
        weights_as_lists = [weight.tolist() for weight in self.weights]

        with open('weights.json', 'w') as f:
            json.dump(weights_as_lists, f)

    def evaluate(self, input):
        """
        Goes throughout forward-popagating proces including input for neural network

        :param input: Input for neural network, should be the same size as first layer of network 
        :return: Output layer 
        """
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

                if layer_index ==(len(self.neurons) -1) and self.output_act_func != "tanh":
                    self.neurons[layer_index] = np.exp(self.neurons[layer_index]) / np.sum(np.exp(self.neurons[layer_index]), dtype=np.float64)
                else:
                    self.neurons[layer_index] = np.tanh(self.neurons[layer_index])
                
        return self.neurons[-1]
    
    def back_propagation(self, answer, learing_rate =0.1):
        """
        Goes throughout back-popagation proces including.
        It's only one interation in witch network is corrected in relation to answer.
        This method should be use only after "evaluate"

        :param answer: Answer as np array for the pervous evaluation
        :param learing_rate: Should be between 0-1
        """

        if len(answer) != len(self.neurons[-1]):
            print("Error: Output size does not match network output size.")
            return

        deltas = [None] * len(self.network_size)

        # Calculate the delta for the output layer
        if self.output_act_func == "tanh":
            output_layer = len(self.network_size) - 1
            deltas[output_layer] = 2 * (self.neurons[output_layer] - answer) * (1 - np.tanh(self.neurons[output_layer]) ** 2)
            # deltas = cost_function'(a) * activation_function'(a) 

        else:
            # Softmax version
            output_layer = len(self.network_size) - 1
            softmax_output = np.exp(self.neurons[output_layer])
            softmax_output /= np.sum(softmax_output)
            deltas[output_layer] = softmax_output - answer
            print("A")


        # Backpropagate the deltas to hidden layers
        for layer in range(output_layer - 1, 0, -1):
            deltas[layer] = np.dot(deltas[layer + 1], self.weights[layer + 1].T) * (1 - np.tanh(self.neurons[layer]) ** 2)
            

        # Update weights and biases
        for layer in range(1, len(self.network_size)):
            self.biases[layer] -= learing_rate * np.sum(deltas[layer], axis=0)
            self.weights[layer] -= learing_rate * np.outer(self.neurons[layer - 1], deltas[layer])

        cost = np.sum((answer - self.neurons[-1]) ** 2)
        self.cost_polt.append(cost)
    
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
    # print(f"For input [3,4]: {network.evaluate([3,4])} \n")
    # network.back_propagation(np.array([1, 0]))
    # network.print_network_status()
    
    network = Network(network_size=[2, 1])
    network.mount()
    # AND expample
    network.train([
        [[1, 1], [1]], 
        [[1, 0], [0]],
        [[0, 1], [0]],
        [[0, 0], [0]]
        ], n_eval = 200000, learning_rate = 0.00001)
    
    print(network.evaluate([1,1]))
    print(network.evaluate([1,0]))

    network.plot_cost()

    network.save()