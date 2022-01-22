from Perceptron import Multilayer_Perceptron, Activation_Functions, Cost_Functions
from matrix import Matrix, pooling, Flags
from simple_feature_extraction import image_to_matrix, get_difference_frame_of_image
import cv2

PATH = './input_images/'
END = '_Balken/1_DONE.jpg'

kernel = Matrix(values=[[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])

<<<<<<< HEAD
images = ['2x4_L', '3er', '3x3_T', '3x3x7_J']

classifications = []
=======
original_image = cv2.imread('./input_images/3er_Balken/0_DONE.jpg')
>>>>>>> e25cd27b507c78b9de1ea54034cccd19d895e96f

for a in images:
    classifications.append([1 if '2x4' in a else 0, 1 if '3er' in a else 0, 1 if '3x3_' in a else 0, 1 if '3x3x7' in a else 0, 1 if '3x5' in a else 0, 1 if '3x7' in a else 0, 1 if '4x4' in a else 0, 1 if '4x6' in a else 0])

original_image = cv2.imread('./input_images/3er_Balken/1_DONE.jpg')

images = [pooling(pooling(get_difference_frame_of_image(kernel, image_to_matrix(cv2.imread(f"{PATH}{image}{END}"))), (2,2), Flags.MAX_POOLING), (2,2), Flags.MAX_POOLING).replace_between(-1, 10, 0, True).replace_between(0.1, 255, 1, True) for image in images]

NN_PATH = './Multilayer_Perceptrons/1.txt'

<<<<<<< HEAD
nn = Multilayer_Perceptron([1000, 400, 400, 200, 100, 8], 80*60, [Activation_Functions.SIGMOID], Cost_Functions.SQUARED_DIFF, 1, NN_PATH, False)
=======
nn = Multilayer_Perceptron([1000, 400, 400, 200, 100, 8], 80*60, [Activation_Functions.SIGMOID], Cost_Functions.CROSS_ENTROPY, 1, NN_PATH, True)
>>>>>>> e25cd27b507c78b9de1ea54034cccd19d895e96f

count = 0
result = []
earlier_result = []
for i in range(2):
    earlier_result = result
<<<<<<< HEAD
    nn.backpropagate([[image.values[a][b] for a in range(len(image.values)) for b in range(len(image.values[a]))] for image in images], classifications, 1)
    for image in images:
        result = nn.get_everything_from_calculate([image.values[a][b] for a in range(len(image.values)) for b in range(len(image.values[a]))])
        stuff = ["2x4", "3er", "3x3", "3x3x7", "3x5", "3x7", "4x4", "4x6"]

        highest_index = 0
        highest_value = 0

        for i in range(len(result)):
            if result[-1][i] > highest_value:
                highest_value = result[-1][i]
                highest_index = i

        print(result[-1])
        print(result[-2])
        print(stuff)
        print(stuff[highest_index])
=======
    nn.backpropagate([[image.values[a][b] for a in range(len(image.values)) for b in range(len(image.values[a]))]], [[0,1,0,0,0,0,0,0]], 1)
    result = nn.calculate([image.values[a][b] for a in range(len(image.values)) for b in range(len(image.values[a]))])
    stuff = ["2x4", "3er", "3x3", "3x3x7", "3x5", "3x7", "4x4", "4x6"]

    highest_index = 0
    highest_value = 0

    for i in range(len(result)):
        if result[i] > highest_value:
            highest_value = result[i]
            highest_index = i

    print(result)
    print(stuff)
    print(stuff[highest_index])

print(f'\n{[result[i] - earlier_result[i] for i in range(len(result))]}')
>>>>>>> e25cd27b507c78b9de1ea54034cccd19d895e96f
