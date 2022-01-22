from Perceptron import Multilayer_Perceptron, Activation_Functions, Cost_Functions

OR_LIST = [[0,1], [1,0], [1,1], [0,0]]
CLASSIFICATIONS = [[1, 0], [1, 0], [0, 1], [0, 1]]

nn = Multilayer_Perceptron([2, 1, 2], 2, [Activation_Functions.SIGMOID, Activation_Functions.SIGMOID, Activation_Functions.SOFTMAX], Cost_Functions.SQUARED_DIFF, 0.001, './test.txt', True)

# for l in nn.layers:
#     for p in l.perceptrons:
#         print(p.weights)

nn.backpropagate(OR_LIST, CLASSIFICATIONS, 1)

# for l in nn.layers:
#     for p in l.perceptrons:
#         print(p.weights)

for l in OR_LIST:
    print(nn.calculate(l))