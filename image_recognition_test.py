from re import TEMPLATE
from Perceptron import Multilayer_Perceptron, Activation_Functions, Cost_Functions
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio

OR_LIST = [[0,1], [1, 0], [1,1], [0,0]]
CLASSIFICATIONS = [[1, 0], [1, 0], [1, 0], [0, 1]]
SINGLE_CLASSIFICATIONS = [[0],[0],[0],[1]]
TEMPLATE = "plotly_dark"

pio.templates.default = TEMPLATE

nn = Multilayer_Perceptron([2, 2], 2, [Activation_Functions.LEAKY_RELU], Cost_Functions.SQUARED_DIFF, 0.5, './test.txt', False)

old_results=[[]]
bias = []
weights = []

for l in nn.layers:
    bias.append([])
    weights.append([])
    for p in l.perceptrons:
        bias[-1].append([])
        weights[-1].append([])
        bias[-1][-1].append(p.bias)
        for w in p.weights:
            weights[-1][-1].append([])
            weights[-1][-1][-1].append(w)

for l in OR_LIST:
    old_results[-1].append(nn.calculate(l))

for i in range(450):
    nn.backpropagate(OR_LIST, CLASSIFICATIONS, 1, False)

    old_results.append([])
    for l in OR_LIST:
        old_results[-1].append(nn.calculate(l))
    
    for l in range(len(nn.layers)):
        for p in range(len(nn.layers[l].perceptrons)):
            bias[l][p].append(nn.layers[l].perceptrons[p].bias)
            for w in range(len(nn.layers[l].perceptrons[p].weights)):
                weights[l][p][w].append(nn.layers[l].perceptrons[p].weights[w])
    
    # for l in nn.layers:
    #     for p in l.perceptrons:
    #         print(f"Bias: {p.bias}\nWeights: {p.weights}\n")
result_names = [f"{i+1}. Perceptron Calculations" for i in range(len(old_results[0][0]))]
bias_names = [f"{(j*2)+1+i}. Perceptron Bias" for j in range(len(bias)) for i in range(len(bias[j]))]
weight_names = [f"{(j*2)+1+i}. Perceptron Weights" for j in range(len(bias)) for i in range(len(bias[j]))]
while len(result_names) < len(bias_names):
    result_names.append('')
names = result_names
names.extend(bias_names)
names.extend(weight_names)

fig = make_subplots(rows=3, cols=len(old_results[0][0]) if len(old_results[0][0]) > sum([len(bias[j]) for j in range(len(bias))]) else sum([len(bias[j]) for j in range(len(bias))]), subplot_titles=names, shared_yaxes=True, shared_xaxes=True)

xs = [i for i in range(len(old_results))]

for i in range(len(old_results[0])):
    for j in range(len(old_results[0][i])):
        fig.add_trace(go.Scatter(x=xs, y=[k[i][j] for k in old_results], name=f"{j+1}: {str(OR_LIST[i])}"), row=1, col=j+1)

for i in range(len(bias)):
    for j in range(len(bias[i])):
        fig.add_trace(go.Scatter(x=xs, y = bias[i][j], name=f"({(i*2)+j+1})     Bias"), row=2, col=i*2+j+1)
        for k in range(len(weights[i][j])):
            fig.add_trace(go.Scatter(x=xs, y = weights[i][j][k], name=f"({(i*2)+j+1})     {k+1}. Weight"), row=3, col=i*2+j+1)

fig.update_xaxes(title_text="Backpropagation Iterations", row=3)
fig.update_yaxes(title_text="Value", col=1)

fig.update_layout(height=900, width=1800, title_text="Debug Stats")
fig.show()