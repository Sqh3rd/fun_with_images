from Perceptron import Multilayer_Perceptron, Activation_Functions, Cost_Functions
from matrix import Matrix, pooling, Flags
from simple_feature_extraction import image_to_matrix, get_difference_frame_of_image
import cv2

PATH = './input_images/'
END = '_Balken/0_DONE.jpg'

kernel = Matrix(values=[[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])

images = ['2x4_L', '3er', '3x3_T', '3x3x7_J']

classifications = []

correct_value = 1

for a in images:
    classifications.append([correct_value if '2x4' in a else 0, correct_value if '3er' in a else 0, correct_value if '3x3_' in a else 0, correct_value if '3x3x7' in a else 0, correct_value if '3x5' in a else 0, correct_value if '3x7' in a else 0, correct_value if '4x4' in a else 0, correct_value if '4x6' in a else 0])

images = [pooling(pooling(get_difference_frame_of_image(kernel, image_to_matrix(cv2.imread(f"{PATH}{image}{END}"))), (2,2), Flags.MAX_POOLING), (2,2), Flags.MAX_POOLING).replace_between(-1, 10, 0, True).replace_between(0.1, 255, 1, True) for image in images]

NN_PATH = './Multilayer_Perceptrons/1.txt'

nn = Multilayer_Perceptron([500, 200, 100, 8], 80*60, [Activation_Functions.LEAKY_RELOG], Cost_Functions.SQUARED_DIFF, 0.01, NN_PATH, False)

count = 0
result = []
earlier_results = [[[0 for j in range(8)] for i in range(4)]]
current_weights = [0 for i in range(8)]
for i in range(100):
    earlier_results.append([])
    earlier_weights = current_weights
    for k, image in enumerate(images):
        all_results = nn.get_everything_from_calculate([image.values[a][b] for a in range(len(image.values)) for b in range(len(image.values[a]))])
        result = all_results[-1]
        stuff = ["2x4", "3er", "3x3", "3x3x7", "3x5", "3x7", "4x4", "4x6"]

        earlier_results[-1].append(result)

        highest_index = 0
        highest_value = 0

        for i in range(len(result)):
            if result[i] > highest_value:
                highest_value = result[i]
                highest_index = i

        print(result)
        print(*[sum(i) for i in all_results[::-1]], sep=', ')
        print(stuff[highest_index])
        print()
    print("Changes to Result:")
    print(*[[earlier_results[-1][k][l] - earlier_results[-2][k][l] for l in range(len(earlier_results[-1][k]))] for k in range(len(earlier_results[-1]))], sep='\n')
    print("Weights:")
    current_weights = [sum([sum(p.weights) for p in l.perceptrons]) for l in nn.layers[::-1]]
    print(*current_weights, sep=', ')
    print("Changes to Weight:")
    print([current_weights[i] - earlier_weights[i] for i in range(len(current_weights))])
    nn.backpropagate([[image.values[a][b] for a in range(len(image.values)) for b in range(len(image.values[a]))] for image in images], classifications, 1)
