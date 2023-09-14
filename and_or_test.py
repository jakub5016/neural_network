from network import Network


# OR Example
def OR_test():
    neural_network = Network(network_size=[2,2])
    neural_network.mount()
    
    data = [[[1,0], [1,0]], [[0,1], [1,0]], [[0,0], [0,0]], [[1,1], [1,1]]]

    neural_network.train(data, learning_rate=0.1, n_eval=100)

    print("EXAMPLE for [1,0]",neural_network.evaluate([1,0]))

    print("EXAMPLE for [0,1]",neural_network.evaluate([0,1]))

    print("EXAMPLE for [0,0]",neural_network.evaluate([0,0]))

    print("EXAMPLE for [1,1]",neural_network.evaluate([1,1]))

    
    neural_network.print_network_status()


    neural_network.plot_cost()

# AND example
def AND_test():
    neural_network = Network(network_size=[2,1])
    neural_network.mount()

    data = [[[1,0], [0]], [[0,1], [0]], [[0,0], [0]], [[1,1], [1]]]

    neural_network.train(data, learning_rate=0.01, n_eval=1000)

    print("EXAMPLE for [1,0]",neural_network.evaluate([1,0]))

    print("EXAMPLE for [0,1]",neural_network.evaluate([0,1]))

    print("EXAMPLE for [0,0]",neural_network.evaluate([0,0]))

    print("EXAMPLE for [1,1]",neural_network.evaluate([1,1]))

    neural_network.print_network_status()

    neural_network.plot_cost()

OR_test()
AND_test()
