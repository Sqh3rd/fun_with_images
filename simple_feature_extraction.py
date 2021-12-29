import cv2
from time import sleep, time
from matrix import Matrix, kernel_multiplicate, pooling, Flags
import numpy as np
from perceptron import Multilayer_Perceptron, Activation_Functions, Cost_Functions
import os

def save_filtered_images_of_image(kernel:Matrix, filters:Matrix, frame:Matrix, out_dir:str):
    for f in frame:
        if len(f.values) % 2 != 0:
            f.insert_line(0)
        if len(f.values[0]) % 2 != 0:
            f.insert_column(0)

    kernel_multiplicated_image = [kernel_multiplicate(first_matrix=kernel, second_matrix=f, stride_length=2, crop_to_val=255, keep_size=True) for f in frame]

    for k in kernel_multiplicated_image:
        if len(k.values) % 2 != 0:
            k.insert_line(0)
        if len(k.values[0]) % 2 != 0:
            k.insert_column(0)

    # cv2.imwrite("./images/old_image.jpg", np.array([[[f.values[a][b] for f in frame] for b in range(len(frame[0].values[a]))] for a in range(len(frame[0].values))]))
    # cv2.imwrite("./images/new_image.jpg", np.array([[[k.values[a][b] for k in kernel_multiplicated_image] for b in range(len(kernel_multiplicated_image[0].values[a]))] for a in range(len(kernel_multiplicated_image[0].values))]))

    difference_frame = [kernel_multiplicate(first_matrix=Matrix(values=[[1,1,1],[1,1,1],[1,1,1]]), second_matrix=k, stride_length=1, crop_to_val=255, get_average=True, get_difference=True, keep_size=True) for k in kernel_multiplicated_image]

    sum_matrix = Matrix(lines=difference_frame[0].lines(), columns=difference_frame[0].columns())

    for diff in difference_frame:
        sum_matrix += diff

    sum_matrix.crop_to_value(255)

    sum_matrix.replace_multiple([i for i in range(min_diff_val)], -1)
    sum_matrix.replace_multiple([i for i in range(min_diff_val, 251)], 1)

    difference_frame = sum_matrix

    newer_frame = np.array(difference_frame.values)

    # cv2.imwrite(out_dir, newer_frame)

    new_images_filtered = []

    for i, filter in enumerate(filters):
        s = time()
        new_image_filtered = kernel_multiplicate(first_matrix=filter, second_matrix=difference_frame, stride_length=1, crop_to_val=255, get_average=True, keep_size=True)
        new_image_filtered = pooling(new_image_filtered, (2,2), Flags.MAX_POOLING)
        new_images_filtered.append(new_image_filtered.values)
        e = time()
        print(f'/\-Filter {i}\n--|{e-s}s\n{int((e-s)//60)}m {int((e-s)%60)}s\n')
        cv2.imwrite(f"{out_dir}/{i}.jpg", np.array(new_image_filtered.values))

min_diff_val = 1

number_of_images = 10

get_image_from_vid = True
if not get_image_from_vid:
    PATH='./input_images/p1/'
    FILETYPE = '.jpg'

    file_paths = []
    for i in range(number_of_images):
        file_paths.append(f'{PATH}{i}{FILETYPE}')

    file_index = 0
else:
    vid = cv2.VideoCapture(0)

kernel = Matrix(values=[[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
filters = [
    Matrix(values=[[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]),
    Matrix(values=[[-1, 1, -1], [-1, 1, -1], [-1, 1, -1]]),
    Matrix(values=[[1, -1, 1], [-1, 1, -1], [1, -1, 1]]),
    Matrix(values=[[-1, -1, 1], [-1, 1, -1], [1, -1, -1]]),
    Matrix(values=[[-1, 1, -1], [1, -1, -1], [-1, 1, -1]]),
    Matrix(values=[[-1, 1, -1], [-1, -1, 1], [-1, 1, -1]]),
    Matrix(values=[[-1, -1, 1], [-1, 1, -1], [-1, 1, -1]]),
    Matrix(values=[[1, -1, -1], [-1, 1, -1], [-1, 1, -1]]),
    Matrix(values=[[-1, 1, -1], [-1, 1, -1], [1, -1, -1]]),
    Matrix(values=[[-1, 1, -1], [-1, 1, -1], [-1, -1, 1]]),
    Matrix(values=[[-1, -1, -1], [-1, -1, 1], [1, 1, -1]]),
    Matrix(values=[[1, -1, -1], [-1, 1, 1], [-1, -1, -1]]),
    Matrix(values=[[-1, -1, -1], [1, 1, -1], [-1, -1, 1]]),
    Matrix(values=[[-1, -1, -1], [-1, 1, 1], [1, -1, -1]]),
    Matrix(values=[[-1, -1, -1], [1, 1, 1], [-1, -1, -1]])
]

is_done = False

while not is_done:
    if get_image_from_vid:
        check, frame = vid.read()
    else:
        frame = cv2.imread(file_paths[file_index], 0)
        file_index += 1

    start = time()

    if len(frame.shape) == 3:
        frame = [Matrix(values = [[frame[a][b][i] for b in range(frame.shape[1])] for a in range(frame.shape[0])]) for i in range(frame.shape[2])]
    else:
        frame = [Matrix(values=frame)]

        
    is_done = True

if get_image_from_vid:
    vid.release()

cv2.destroyAllWindows()

end = time()

print(end - start)