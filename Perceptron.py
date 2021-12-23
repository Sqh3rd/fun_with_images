from random import randint
from matrix import Matrix
from math import exp

class Activation_Functions:
    def sigmoid(inp:float) -> float:
        return 1/(1+exp(-inp))

    def relu(inp:float) -> float:
        return max(0, inp)

    def leaky_relu(inp:float, a) -> float:
        if inp > 0:
            return max(0,inp)
        else:
            return a * inp
    
    def softmax(inp:list) -> list:
        return [i/sum(inp) for i in inp]

    SIGMOID = sigmoid
    RELU = relu
    LEAKY_RELU = leaky_relu
    SOFTMAX = softmax

class Cost_Functions:
    def squared_diff(pred:float, act:float):
        return (pred-act)**2
    
    SQUARED_DIFF = squared_diff

class Perceptron:
    def __init__(self, size) -> None:
        self.weights = [randint(0,20) for i in range(size)]
        self.bias = randint(0,20)

    def weighted_sum(self, inp:list) -> float:
        self.sum = 0
        for i in range(len(inp)):
            self.sum += inp[i] * self.weights[i]
        self.sum += self.bias
        return self.sum

class Layer:
    def __init__(self, size, size_perceptron, activation_function, leaky_relu_const = 0.0001):
        self.leaky_relu_const = leaky_relu_const
        self.perceptrons = []
        for i in range(size):
            self.perceptrons.append(Perceptron(size_perceptron))
        self.activation_function = activation_function
    
    def calculate(self, inp:list) -> list:
        if self.activation_function == Activation_Functions.SOFTMAX:
            end_list = self.activation_function([p.weighted_sum(inp) for p in self.perceptrons])
        else:
            end_list = []
            for perceptron in self.perceptrons:
                if self.activation_function == Activation_Functions.LEAKY_RELU:
                    end_list.append(self.activation_function(perceptron.weighted_sum(inp), self.leaky_relu_const))
                else:
                    end_list.append(self.activation_function(perceptron.weighted_sum(inp)))
        return end_list

if __name__ == '__main__':
    l = Layer(5, 20, Activation_Functions.LEAKY_RELU)
    a = Layer(3, 20, Activation_Functions.SOFTMAX)
    m = [1,2,3,4,5,6,7,8,9]
    e = l.calculate(m)
    f = a.calculate(e)
    print(e)
    print(f)
