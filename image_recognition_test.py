from Perceptron import Multilayer_Perceptron, Activation_Functions, Cost_Functions

OR_LIST = [[0,1], [1, 0], [1,1], [0,0]]
CLASSIFICATIONS = [[1, 0], [1, 0], [0, 1], [0, 1]]

nn = Multilayer_Perceptron([2, 2], 2, [Activation_Functions.LEAKY_RELU], Cost_Functions.SQUARED_DIFF, 0.1, './test.txt', False)

# for l in nn.layers:
#     for p in l.perceptrons:
#         print(p.weights)

nn.backpropagate(OR_LIST, CLASSIFICATIONS, 100)

for l in nn.layers:
    for p in l.perceptrons:
        print(f"Bias: {p.bias}\nWeights: {p.weights}\n")

for l in OR_LIST:
    print(nn.calculate(l))