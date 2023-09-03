import numpy as np
from random import random

class Neuron():
    def __init__(self, next_layer_size = 0, lenght_stack = None):
        self.next_layer_size = next_layer_size
        self.weights = np.empty(next_layer_size)

        if next_layer_size > 0:
           self.next_layer_neurons = [Neuron()] * self.next_layer_size

        else:
            self.next_layer_neurons = None

    def mount(self):
        for i in self.weights:
            i = random()

        for i in self.next_layer_neurons:
            i.mount()

class Stack():
    def __init__(self, init_table = []):
        self.table = np.array(init_table)

    def pop(self):
        last_element_index = len(self.table) -1
        element = self.table[last_element_index]
        self.table = np.delete(self.table, last_element_index)
        return element
    
if __name__ == "__main__":
    network_size = Stack([2, 2, 3])
    print(network_size.table)
    var = network_size.pop()
    print(var)
    print(network_size.table)
    
    table = [Neuron(2)] * 256