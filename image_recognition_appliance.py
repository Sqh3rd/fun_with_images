from perceptron import Multilayer_Perceptron
from matrix import Matrix, pooling, Flags
from simple_feature_extraction import image_to_matrix, get_difference_frame_of_image
import cv2

kernel = Matrix(values=[[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])

original_image = cv2.imread('./input_images/3x7_J_Balken/0_DONE.jpg')

image = pooling(pooling(get_difference_frame_of_image(kernel, image_to_matrix(original_image)), (2,2), Flags.MAX_POOLING), (2,2), Flags.MAX_POOLING)

NN_PATH = './Multilayer_Perceptrons/0.txt'

nn = Multilayer_Perceptron([], 0, [], 0, 0, NN_PATH, True)

count = 0

for l in nn.layers:
    for p in l.perceptrons:
        for w in p.weights:
            if w == 0:
                count += 1
        print(f'\r{count}', end='')
print('\n')

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