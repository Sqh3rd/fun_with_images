from perceptron import Multilayer_Perceptron, Activation_Functions, Cost_Functions
from matrix import Matrix, pooling, Flags
from simple_feature_extraction import image_to_matrix, get_difference_frame_of_image
import cv2

kernel = Matrix(values=[[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])

original_image = cv2.imread('./input_images/2x4_L_Balken/0_DONE.jpg')

image = pooling(pooling(get_difference_frame_of_image(kernel, image_to_matrix(original_image)), (2,2), Flags.MAX_POOLING), (2,2), Flags.MAX_POOLING)

NN_PATH = './Multilayer_Perceptrons/1.txt'

nn = Multilayer_Perceptron([1000, 400, 400, 200, 100, 8], 80*60, [Activation_Functions.SIGMOID], Cost_Functions.SQUARED_DIFF, 0.01, NN_PATH)

count = 0
result = []
earlier_result = []
for i in range(2):
    earlier_result = result
    nn.backpropagate([[image.values[a][b] for a in range(len(image.values)) for b in range(len(image.values[a]))]], [[1,0,0,0,0,0,0,0]], 1)
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