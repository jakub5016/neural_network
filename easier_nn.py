from random import random
import numpy as np 
import matplotlib
matplotlib.use("TkAgg")
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
            self.neurons.append(np.zeros(item))

            if index > 0:
                self.weights.append(np.random.normal(size=(self.network_size[index-1], item)))
                self.biases.append(np.zeros(item))
            else:
                self.weights.append(None)
                self.biases.append(None)

        for index, item in enumerate(self.neurons):
            if index > 0:
                item = np.matmul(self.neurons[index-1], self.weights[index])

        self.print_network_status()


    def print_network_status(self):
        print("\nNEURONS:", self.neurons)
        print("WEIGHTS: ", self.weights)
        print("BIASES:", self.biases, "\n")

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
            if layer_index > 0: # Starting layer don't have any val before 
                self.neurons[layer_index] = np.matmul(self.neurons[layer_index-1], self.weights[layer_index]) # Multiply
                
                for elem_index in range(len(self.neurons[layer_index])): #Add biases
                    self.neurons[layer_index][elem_index] += self.biases[layer_index][elem_index]
                    
                if layer_index !=(len(self.neurons) -1):
                    self.neurons[layer_index] =  self.neurons[layer_index] * (self.neurons[layer_index] > 0)  # ReLU activation
                else:
                    self.neurons[layer_index] = np.tanh(self.neurons[layer_index]) #Tanh activation
                
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

        output_layer = len(self.network_size) - 1

        # Calculate the delta for the output layer
        deltas[output_layer] = 2 * (self.neurons[output_layer] - answer) * ((1/np.cosh(network.neurons[-1]))**2)
            # deltas = cost_function'(a) * activation_function'(a) 


        # Backpropagate the deltas to hidden layers
        for layer in range(output_layer - 1, 0, -1):
            deltas[layer] = np.dot(deltas[layer + 1], self.weights[layer + 1].T) * (1 - np.tanh(self.neurons[layer]) ** 2)
            

        # Update weights and biases
        for layer in range(1, len(self.network_size)):
            self.biases[layer] -= learing_rate * np.sum(deltas[layer], axis=0)
            self.weights[layer] -= learing_rate * np.outer(self.neurons[layer - 1], deltas[layer])

        cost = np.sum((answer - self.neurons[-1]) ** 2)
        self.cost_polt.append(cost)

if __name__ == "__main__":
    # network = Network(network_size=[2, 2, 2])
    # network.mount()
    # print(f"For input [3,4]: {network.evaluate([3,4])} \n")
    # network.back_propagation(np.array([1, 0]))
    # network.print_network_status()
    
    network = Network(network_size=[2,1])
    network.mount()

    print("OUTPUT: ",network.evaluate([1,0]))
    network.print_network_status()
    # XOR
    # data = [[[1,0], [0]], [[0,1], [0]], [[0,0], [0]], [[1,1], [1]]]
    # network.train(data, n_eval=100)
    
    # print("EXAMPLE for [1,0]",network.evaluate([1,0]))

    # print("EXAMPLE for [0,1]",network.evaluate([0,1]))

    # print("EXAMPLE for [0,0]",network.evaluate([0,0]))

    # print("EXAMPLE for [1,1]",network.evaluate([1,1]))

    # network.plot_cost()