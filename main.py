import numpy as np
from random import random

class Stack():
    def __init__(self, init_table = []):
        self.table = np.array(init_table)

    def pop(self):
        try:
            last_element_index = len(self.table) -1
            element = self.table[last_element_index]
            self.table = np.delete(self.table, last_element_index)
            return element
        except:
            return 0 # Just for next_layer_size porpuse in Neuron __init__
    


class Neuron():
    def __init__(self, lenght_stack = Stack(0)):
        self.next_layer_size = lenght_stack.pop()

        if self.next_layer_size > 0:
            self.next_layer_neurons = [Neuron(lenght_stack)] * self.next_layer_size
            self.weights = np.empty(self.next_layer_size)
           
            for i in self.weights:
                i = random()

        else:
            self.next_layer_neurons = None
