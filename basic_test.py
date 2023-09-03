from main import Stack, Neuron

def test_stack():
    network_size = Stack([2, 2, 3])
    assert  network_size.pop() == 3 and len(network_size.table)==2


def test_neuron():
    network_size = Stack([2, 2, 3])
    table = [Neuron(network_size)] * 256