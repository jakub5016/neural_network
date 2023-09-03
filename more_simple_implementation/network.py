import numpy as np
from random import random


network_size =  [2, 2, 3]

neurons = np.empty([len(network_size), max(network_size)]) 

for index, elem in enumerate(neurons):
    print(network_size[index])
    elem[network_size[index]:] = 0


print(neurons)




