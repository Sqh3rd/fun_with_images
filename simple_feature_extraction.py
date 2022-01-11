import cv2
from matrix import Matrix, kernel_multiplicate, pooling, Flags
import numpy as np
import os

BLACKLIST = ['5er_Balken', '7er_Balken', '9er_Balken', '11er_Balken', '13er_Balken', '15er_Balken', '4x6_J_Balken']

min_diff_val = 1

PATH='./input_images/'
FILETYPE = '.jpg'

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


#Applies the given kernel on the given image and returns a difference frame
def get_difference_frame_of_image(kernel:Matrix, frame:list[Matrix]) -> Matrix:
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

    difference_frame = [kernel_multiplicate(first_matrix=Matrix(values=[[1,1,1],[1,1,1],[1,1,1]]), second_matrix=k, stride_length=1, crop_to_val=255, get_average=True, get_difference=True, keep_size=True) for k in kernel_multiplicated_image]

    sum_matrix = Matrix(lines=difference_frame[0].lines(), columns=difference_frame[0].columns())

    for diff in difference_frame:
        sum_matrix += diff

    sum_matrix.crop_to_value(255)

    sum_matrix.replace_multiple([i for i in range(min_diff_val)], -1)
    sum_matrix.replace_multiple([i for i in range(min_diff_val, 256)], 1)

    difference_frame = sum_matrix

    return difference_frame

#Applies every given filter to a given image and saves all resulting images in a given directory
def save_filtered_images_of_image(filters:Matrix, image:Matrix, out_dir:str) -> None:

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    new_images_filtered = []
    for i, filter in enumerate(filters):
        new_image_filtered = kernel_multiplicate(first_matrix=filter, second_matrix=image, stride_length=1, crop_to_val=255, get_average=True, keep_size=True)
        new_image_filtered = pooling(new_image_filtered, (2,2), Flags.MAX_POOLING)
        new_images_filtered.append(new_image_filtered.values)
        out_file = f'{out_dir}/{i}.jpg'
        cv2.imwrite(out_file, np.array(new_image_filtered.values))

def image_to_matrix(frame) -> list[Matrix]:
    if len(frame.shape) == 3:
        out = [Matrix(values = [[frame[a][b][i] for b in range(frame.shape[1])] for a in range(frame.shape[0])]) for i in range(frame.shape[2])]
    else:
        out = [Matrix(values=frame)]
    return out

if __name__ == "__main__":
    for o in os.listdir(PATH):
        index = 0
        images = []
        if o in BLACKLIST:
            continue
        for i, a in enumerate(os.listdir(f'{PATH}{o}')[:(len(os.listdir(f'{PATH}{o}'))//10)*2]):
            if not '.' in a or "READY" in a:
                continue
            if 'DONE' in a:
                index += 1
                continue
            images.append(a)
            break

        for a in images:
            fr = cv2.imread(f'{PATH}{o}/{a}')
            cv2.imwrite(f'{PATH}{o}/{index}_READY.jpg', np.array(pooling(pooling(get_difference_frame_of_image(kernel, image_to_matrix(fr)), (2,2), Flags.MAX_POOLING), (2,2), Flags.MAX_POOLING).values))
            os.rename(f'{PATH}{o}/{a}', f'{PATH}{o}/{index}_DONE.jpg')
            index += 1