from Perceptron import Cost_Functions, Multilayer_Perceptron, Activation_Functions
from simple_feature_extraction import image_to_matrix, get_difference_frame_of_image
from matrix import Matrix, pooling, Flags
import cv2
import os
from random import randint

PATH='./input_images'
NN_PATH = './Multilayer_Perceptrons/0.txt'

BLACKLIST = ['5er_Balken', '7er_Balken', '9er_Balken', '11er_Balken', '13er_Balken', '15er_Balken']

should_read = False

nn_exists = os.path.isfile(NN_PATH)

possible_list = []

for a in os.listdir(PATH):
    if a in BLACKLIST:
        continue
    for b in os.listdir(f'{PATH}/{a}'):
        if 'DONE' in b:
            path_to_image = f'{PATH}/{a}/{b}'
            classification_of_image = [1 if '2x4' in a else 0, 1 if '3er' in a else 0, 1 if '3x3_' in a else 0, 1 if '3x3x7' in a else 0, 1 if '3x5' in a else 0, 1 if '3x7' in a else 0, 1 if '4x4' in a else 0, 1 if '4x6' in a else 0]
            possible_list.append((path_to_image, classification_of_image))

test_batch_length = 10
test_batches = []
for i in range(int(len(possible_list)//test_batch_length)):
    test_batches.append([])
    for j in range(test_batch_length):
        test_batches[-1].append(possible_list[randint(0, len(possible_list) - 1)])

if should_read and nn_exists:
    nn = Multilayer_Perceptron([1000, 400, 400, 200, 100, 8], 80*60, [Activation_Functions.LEAKY_RELOG], Cost_Functions.SQUARED_DIFF, 10, NN_PATH, True)
else:
    nn = Multilayer_Perceptron([1000, 400, 400, 200, 100, 8], 80*60, [Activation_Functions.LEAKY_RELOG], Cost_Functions.SQUARED_DIFF, 10, NN_PATH)
    nn.write_to_file()

for i, t in enumerate(test_batches):
    images = [pooling(get_difference_frame_of_image(Matrix(values=[[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]]), image_to_matrix(cv2.imread(f'{t[j][0]}'))), (2,2), Flags.MAX_POOLING, 2) for j in range(len(t))]
    nn.backpropagate([[im.values[a][b] for a in range(len(im.values)) for b in range(len(im.values[a]))] for im in images], [t[j][1] for j in range(len(t))], 1)
    print(f"{i+1}/{len(test_batches)}")
    if (i+1) % 7 == 0:
        break