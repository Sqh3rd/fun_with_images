from random import random
from matrix import Matrix
from math import exp, log

class Activation_Functions:
    def sigmoid(self, inp:float) -> float:
        return 1/(1+exp(-inp))

    def der_sigmoid(self, inp:float) -> float:
        return self.sigmoid(Activation_Functions, inp) * (1 - self.sigmoid(Activation_Functions, inp))

    def relu(self, inp:float) -> float:
        return inp if inp > 0 else 0

    def der_relu(self, inp:float) -> float:
        return 1 if inp > 0 else 0

    def leaky_relu(self, inp:float, const) -> float:
        if inp > 0:
            return max(inp, 0)
        else:
            return min(inp*const, 0)

    def der_leaky_relu(self, inp:float, const) -> float:
        if inp > 0:
            return 1
        else:
            return const
    
    def softmax(self, inp:list) -> list:
        return [i/sum([a for a in inp]) for i in inp]

    def der_softmax(self, inp:list, index:int) -> float:
        s = self.softmax(Activation_Functions, inp)
        sum_of_influence = 0
        for i in range(len(inp)):
            if i == index:
                sum_of_influence += s[i] * (1-s[i])
            else:
                sum_of_influence += -s[i] * s[index]
        return sum_of_influence

    SIGMOID = (sigmoid, der_sigmoid)
    RELU = (relu, der_relu)
    LEAKY_RELU = (leaky_relu, der_leaky_relu)
    SOFTMAX = (softmax, der_softmax)

class Cost_Functions:
    def squared_diff(pred:list[float], act:list[float]):
        return [(pred[i] - act[i])**2 for i in range(len(pred))]
    
    def der_squared_diff(pred:float, act:float):
        return 2*(pred - act)

    def cross_entropy(pred:list[float], act:list[float]):
        return [(-act[i]*log(pred[i]) for i in range(len(pred)))]

    def der_cross_entropy(pred:list[float], act:list[float]):
        return -act*(1/pred)
    
    SQUARED_DIFF = (squared_diff, der_squared_diff)
    CROSS_ENTROPY = (cross_entropy, der_cross_entropy)

class Perceptron:
    def __init__(self, size) -> None:
        self.weights = [0.5 for i in range(size)]
        self.bias = 0.5

    def weighted_sum(self, inp:list) -> float:
        self.sum = sum([inp[i] * self.weights[i] for i in range(len(inp))])
        self.sum += self.bias
        return self.sum

class Layer:
    def __init__(self, size, size_perceptron, activation_function, leaky_relu_const = 0.01, cost_function = None):
        self.leaky_relu_const = leaky_relu_const
        self.perceptrons = [Perceptron(size_perceptron) for i in range(size)]
        self.activation_function = activation_function
        self.cost_function = cost_function
    
    def calculate(self, inp:list) -> list:
        if self.activation_function == Activation_Functions.SOFTMAX:
            end_list = self.activation_function[0](Activation_Functions, [perceptron.weighted_sum(inp) for perceptron in self.perceptrons])
        else:
            end_list = []
            for perceptron in self.perceptrons:
                if self.activation_function == Activation_Functions.LEAKY_RELU:
                    end_list.append(self.activation_function[0](Activation_Functions, inp = perceptron.weighted_sum(inp), const = self.leaky_relu_const))
                else:
                    end_list.append(self.activation_function[0](Activation_Functions, perceptron.weighted_sum(inp)))
        return end_list

    def calculate_cost(self, inp:list, act:list) -> list:
        if self.activation_function[0] == Activation_Functions.SOFTMAX[0]:
            end_list = self.activation_function[0](Activation_Functions, [p.weighted_sum(inp) for p in self.perceptrons])
        else:
            end_list = []
            for perceptron in self.perceptrons:
                if self.activation_function[0] == Activation_Functions.LEAKY_RELU[0]:
                    end_list.append(self.activation_function[0](Activation_Functions, inp = perceptron.weighted_sum(inp), const = self.leaky_relu_const))
                else:
                    end_list.append(self.activation_function[0](Activation_Functions, perceptron.weighted_sum(inp)))
        end_cost = self.cost_function[0](end_list, act)
        return end_list, end_cost

class Multilayer_Perceptron:
    def __init__(self, amount_neurons:list, amount_input:int, activation_functions:list, cost_function, step_size:float, file_path:str, read:bool = False, leaky_relu_const = 0.01):
        self.file_path = file_path
        if read:
            self.read_from_file()
            return
        self.step_size = step_size
        self.cost_function = cost_function
        self.activation_functions = activation_functions
        self.amount_inputs = amount_input
        while len(activation_functions) <= len(amount_neurons):
            self.activation_functions.append(activation_functions[-1])   
        # self.activation_functions[len(amount_neurons)] = Activation_Functions.SOFTMAX
        self.layers = []
        for i in range(len(amount_neurons)):
            self.layers.append(Layer(amount_neurons[i], amount_neurons[i-1] if i > 0 else amount_input, self.activation_functions[i], cost_function = self.cost_function, leaky_relu_const=leaky_relu_const))
        # self.layers.append(Layer(amount_neurons[-1], amount_neurons[-1], self.activation_functions[-1], cost_function = self.cost_function))
        self.step_size = step_size
        self.leaky_relu_const = leaky_relu_const
    
    def base_calculation(self, inp:list) -> list:
        result = [inp]
        for i in range(len(self.layers) - 1):
            result.append(self.layers[i].calculate(result[-1]))
        return result

    def calculate(self, inp:list) -> list:
        if len(self.layers) > 1:
            return self.layers[-1].calculate(self.base_calculation(inp)[-1])
        else:
            return self.base_calculation(inp)

    def get_everything_from_calculate(self, inp:list) -> list:
        result = self.base_calculation(inp)
        result.append(self.layers[-1].calculate(result[-1]))
        return result

    def backpropagate(self, inp:list[list], act:list[list], iterations:int, print_current_iteration_and_input=True) -> None:
        for it in range(iterations):
            preds = []

            for i in range(len(inp)):
                preds.append([])
                preds[-1].append(inp[i])
                for layer in self.layers:
                    preds[-1].append(layer.calculate(preds[-1][-1]))

            results = []
            costs = []
            weight_change_suggestions = [[[[[] for i in range(len(inp))] for w in p.weights] for p in l.perceptrons] for l in self.layers]
            bias_change_suggestions = [[[[] for i in range(len(inp))] for p in l.perceptrons] for l in self.layers]
            perceptron_change_suggestions = [[[[] for i in range(len(inp))] for p in l.perceptrons] for l in self.layers[:-1]]
            temp_perceptron_change_suggestions = [[[[] for i in range(len(inp))] for p in l.perceptrons] for l in self.layers]

            for i in range(len(act)):
                for j in range(len(act[i])):
                    temp_perceptron_change_suggestions[-1][j][i] = act[i][j]

            for i in range(len(inp)):
                if print_current_iteration_and_input:
                    print(f'\rIteration {it+1}/{iterations}        Input {i+1}/{len(inp)}{"".join([" " for i in range(20)])}', end='')
                for j in range(1, len(self.layers) + 1):
                    current_layer_index = len(self.layers) - j
                    current_layer = self.layers[current_layer_index]
                    previous_layer = self.layers[current_layer_index - 1]

                    temp = [temp_perceptron_change_suggestions[current_layer_index][k][i] for k in range(len(temp_perceptron_change_suggestions[current_layer_index]))]

                    result, cost = current_layer.calculate_cost(preds[i][current_layer_index], temp)
                    results.append(result)
                    costs.append(cost)
                    for k in range(len(current_layer.perceptrons)):
                        current_perceptron = current_layer.perceptrons[k]
                        current_perceptron_value = preds[i][current_layer_index+1][k]
                        should_be_current_value = temp_perceptron_change_suggestions[current_layer_index][k][i]
                        for l in range(len(preds[i][current_layer_index])):
                            previous_perceptron_value = preds[i][current_layer_index][l]
                            derivative_activation_function = current_layer.activation_function[1]
                            derivative_cost_function = self.cost_function[1]
                            if derivative_activation_function == Activation_Functions.SOFTMAX[1]:
                                weight_change_suggestions[current_layer_index][k][l][i].append(-self.step_size*(previous_perceptron_value*derivative_activation_function(Activation_Functions, [perceptron.sum for perceptron in current_layer.perceptrons], k)*derivative_cost_function(current_perceptron_value, should_be_current_value)))
                            elif derivative_activation_function != Activation_Functions.LEAKY_RELU[1]:
                                weight_change_suggestions[current_layer_index][k][l][i].append(-self.step_size*(previous_perceptron_value*derivative_activation_function(Activation_Functions, current_perceptron.sum)*derivative_cost_function(current_perceptron_value, should_be_current_value)))
                            else:
                                weight_change_suggestions[current_layer_index][k][l][i].append(-self.step_size*(previous_perceptron_value*derivative_activation_function(Activation_Functions, current_perceptron.sum, self.leaky_relu_const)*derivative_cost_function(current_perceptron_value, should_be_current_value)))
                        if derivative_activation_function == Activation_Functions.SOFTMAX[1]:
                            bias_change_suggestions[current_layer_index][k][i].append(-self.step_size*(derivative_activation_function(Activation_Functions, [perceptron.sum for perceptron in current_layer.perceptrons], k)*derivative_cost_function(current_perceptron_value, should_be_current_value)))
                        elif derivative_activation_function != Activation_Functions.LEAKY_RELU[1]:
                            bias_change_suggestions[current_layer_index][k][i].append(-self.step_size*(derivative_activation_function(Activation_Functions, current_perceptron.sum)*derivative_cost_function(current_perceptron_value, should_be_current_value)))
                        else:
                            bias_change_suggestions[current_layer_index][k][i].append(-self.step_size*(derivative_activation_function(Activation_Functions, current_perceptron.sum, self.leaky_relu_const)*derivative_cost_function(current_perceptron_value, should_be_current_value)))

                    if j == len(self.layers):
                        continue

                    #
                    # NOT SO IMPORTANTE
                    #
                    # Shits not working around here probably
                    #
                    # Insert -self.step_size* before sum()
                    # 

                    for k in range(len(previous_layer.perceptrons)):
                        derivative_activation_function = current_layer.activation_function[1]
                        derivative_cost_function = self.cost_function[1]
                        if derivative_activation_function == Activation_Functions.SOFTMAX[1]:
                            perceptron_change_suggestions[current_layer_index - 1][k][i].append(-self.step_size*sum([current_layer.perceptrons[l].weights[k]*derivative_activation_function(Activation_Functions, [perceptron.sum for perceptron in current_layer.perceptrons], k)*derivative_cost_function(preds[i][current_layer_index][k], temp_perceptron_change_suggestions[current_layer_index][l][i]) for l in range(len(current_layer.perceptrons))]))
                        elif derivative_activation_function != Activation_Functions.LEAKY_RELU[1]:
                            perceptron_change_suggestions[current_layer_index - 1][k][i].append(-self.step_size*sum([current_layer.perceptrons[l].weights[k]*derivative_activation_function(Activation_Functions, current_layer.perceptrons[l].sum)*derivative_cost_function(preds[i][current_layer_index][k], temp_perceptron_change_suggestions[current_layer_index][l][i]) for l in range(len(current_layer.perceptrons))]))
                        else:
                            perceptron_change_suggestions[current_layer_index - 1][k][i].append(-self.step_size*sum([current_layer.perceptrons[l].weights[k]*derivative_activation_function(Activation_Functions, current_layer.perceptrons[l].sum, self.leaky_relu_const)*derivative_cost_function(preds[i][current_layer_index][k], temp_perceptron_change_suggestions[current_layer_index][l][i]) for l in range(len(current_layer.perceptrons))]))
                        temp_perceptron_change_suggestions[current_layer_index - 1][k][i] = previous_layer.perceptrons[k].sum + sum(perceptron_change_suggestions[current_layer_index - 1][k][i])

            print(perceptron_change_suggestions)

            for j in range(len(self.layers)):
                for k in range(len(self.layers[j].perceptrons)):
                    self.layers[j].perceptrons[k].bias += sum([a[0] for a in bias_change_suggestions[j][k]])/len([a[0] for a in bias_change_suggestions[j][k]])
                    for l in range(len(self.layers[j].perceptrons[k].weights)):
                        if weight_change_suggestions[j][k][l][0] == []:
                            continue
                        self.layers[j].perceptrons[k].weights[l] += sum([a[0] for a in weight_change_suggestions[j][k][l]])/len([a[0] for a in weight_change_suggestions[j][k][l]])
        if print_current_iteration_and_input:
            print('')
        self.write_to_file(should_print=print_current_iteration_and_input)

    def write_to_file(self, should_print=True):
        with open(self.file_path, 'w') as f:
            match self.cost_function:
                case Cost_Functions.SQUARED_DIFF:
                    f.write('c1\n')
                case Cost_Functions.CROSS_ENTROPY:
                    f.write('c2\n')
            for function in self.activation_functions:
                match function:
                    case Activation_Functions.SIGMOID:
                        f.write('a1\n')
                    case Activation_Functions.RELU:
                        f.write('a2\n')
                    case Activation_Functions.LEAKY_RELU:
                        f.write('a3\n')
                    case Activation_Functions.SOFTMAX:
                        f.write('a4\n')
            f.write(f's{self.step_size}\ni{self.amount_inputs}\nlrc{self.leaky_relu_const}\n')
            for layer in self.layers:
                f.write(f'l{(len(layer.perceptrons))}\n')
                for perceptron in layer.perceptrons:
                    f.write('p\n')
                    f.write(f'b{perceptron.bias}\n')
                    for weight in perceptron.weights:
                        f.write(f'w{weight}\n')
        if should_print:
            print('Writing complete!')

    def read_from_file(self):
        self.activation_functions = []
        self.layers = []
        p_index = -1
        w_index = -1
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                match line[0]:
                    case 'c':
                        match line[1]:
                            case '1':
                                self.cost_function = Cost_Functions.SQUARED_DIFF
                            case '2':
                                self.cost_function = Cost_Functions.CROSS_ENTROPY
                    case 'a':
                        match line[1]:
                            case '1':
                                self.activation_functions.append(Activation_Functions.SIGMOID)
                            case '2':
                                self.activation_functions.append(Activation_Functions.RELU)
                            case '3':
                                self.activation_functions.append(Activation_Functions.LEAKY_RELU)
                            case '4':
                                self.activation_functions.append(Activation_Functions.SOFTMAX)
                    case 'i':
                        self.amount_inputs = int(line[1:])
                    case 's':
                        self.step_size = float(line[1:])
                    case 'l':
                        if line[:3] == 'lrc':
                            self.leaky_relu_const = line[3:]
                        else:
                            self.layers.append(Layer(int(line[1:]), (len(self.layers[-1].perceptrons)) if len(self.layers) > 0 else self.amount_inputs, self.activation_functions[len(self.layers)], cost_function = self.cost_function))
                            p_index = -1
                    case 'p':
                        p_index += 1
                        w_index = 0
                    case 'b':
                        self.layers[-1].perceptrons[p_index].bias = float(line[1:])
                    case 'w':
                        self.layers[-1].perceptrons[p_index].weights[w_index] = float(line[1:])
                        w_index += 1

if __name__ == '__main__':
    l = Multilayer_Perceptron([10,5,4,2], 4, [Activation_Functions.SIGMOID], Cost_Functions.SQUARED_DIFF, 0.01, './test.txt')
    inputs = [[1,2,3,4],[5,6,7,7],[9,10,11,12],[13,14,15,19]]
    actuals = [[0,1],[1,0],[0,1],[1,0]]
    calculation_inputs = [1,2,3,4]
    print(len(calculation_inputs))
    print(l.calculate(calculation_inputs))
    l.backpropagate(inputs, actuals, 30)
    print(l.calculate(calculation_inputs))
    l.write_to_file()
    m = Multilayer_Perceptron([],0,[],0,0,'./test.txt', True)
    print(m.calculate(calculation_inputs))