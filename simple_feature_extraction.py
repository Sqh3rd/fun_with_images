import cv2
from time import sleep, time
from matrix import Matrix, kernel_multiplicate, pooling, Flags
import numpy as np

get_image_from_vid = False
PATH='./input_images/cat_image_3.jpeg'

if get_image_from_vid:
    vid = cv2.VideoCapture(0)

kernel = Matrix(
    values=[[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
filters = [
    #100
    #010
    #001
    Matrix(values=[[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]),
    #010
    #010
    #010
    Matrix(values=[[-1, 1, -1], [-1, 1, -1], [-1, 1, -1]]),
    #101
    #010
    #101
    Matrix(values=[[1, -1, 1], [-1, 1, -1], [1, -1, 1]]),
    #001
    #010
    #100
    Matrix(values=[[-1, -1, 1], [-1, 1, -1], [1, -1, -1]]),
    #010
    #100
    #010
    Matrix(values=[[-1, 1, -1], [1, -1, -1], [-1, 1, -1]]),
    #010
    #001
    #010
    Matrix(values=[[-1, 1, -1], [-1, -1, 1], [-1, 1, -1]]),
    #001
    #010
    #010
    Matrix(values=[[-1, -1, 1], [-1, 1, -1], [-1, 1, -1]]),
    #100
    #010
    #010
    Matrix(values=[[1, -1, -1], [-1, 1, -1], [-1, 1, -1]]),
    #010
    #010
    #100
    Matrix(values=[[-1, 1, -1], [-1, 1, -1], [1, -1, -1]]),
    #010
    #010
    #001
    Matrix(values=[[-1, 1, -1], [-1, 1, -1], [-1, -1, 1]]),
    #000
    #001
    #110
    Matrix(values=[[-1, -1, -1], [-1, -1, 1], [1, 1, -1]]),
    #100
    #011
    #000
    Matrix(values=[[1, -1, -1], [-1, 1, 1], [-1, -1, -1]]),
    #000
    #110
    #001
    Matrix(values=[[-1, -1, -1], [1, 1, -1], [-1, -1, 1]]),
    #000
    #011
    #100
    Matrix(values=[[-1, -1, -1], [-1, 1, 1], [1, -1, -1]]),
    #000
    #111
    #000
    Matrix(values=[[-1, -1, -1], [1, 1, 1], [-1, -1, -1]]),
]

is_done = False

while not is_done:
    if get_image_from_vid:
        check, frame = vid.read()
    else:
        frame = cv2.imread(PATH, 0)

    start = time()

    print(frame.shape)

    if len(frame.shape) == 3:
        frame = [Matrix(values = [[frame[a][b][i] for b in range(frame.shape[1])] for a in range(frame.shape[0])]) for i in range(frame.shape[2])]
    else:
        frame = [Matrix(values=frame)]

    print(repr(frame))
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

    old_image = [[[]]]

    cv2.imwrite("./images/old_image.jpg", np.array([[[f.values[a][b] for f in frame] for b in range(len(frame[0].values[a]))] for a in range(len(frame[0].values))]))
    cv2.imwrite("./images/new_image.jpg", np.array([[[k.values[a][b] for k in kernel_multiplicated_image] for b in range(len(kernel_multiplicated_image[0].values[a]))] for a in range(len(kernel_multiplicated_image[0].values))]))

    difference_frame = [kernel_multiplicate(first_matrix=Matrix(values=[[1,1,1],[1,1,1],[1,1,1]]), second_matrix=k, stride_length=1, crop_to_val=255, get_average=True, get_difference=True, keep_size=True) for k in kernel_multiplicated_image]

    sum_matrix = Matrix(lines=difference_frame[0].lines(), columns=difference_frame[0].columns())

    for diff in difference_frame:
        sum_matrix += diff

    difference_frame = sum_matrix

    difference_frame.crop_to_value(255)

    newer_frame = np.array(difference_frame.values)

    cv2.imwrite("./images/difference_frame.jpg", newer_frame)

    new_images_filtered = []

    i = 0
    for filter in filters:
        new_image_filtered = kernel_multiplicate(first_matrix=filter, second_matrix=difference_frame, stride_length=1, crop_to_val=255, get_average=True, keep_size=True)
        new_image_filtered = pooling(new_image_filtered, (2,2), Flags.MAX_POOLING)
        new_images_filtered.append(new_image_filtered.values)
        cv2.imwrite(f"./images/filtered_images/filter_{i}.jpg", np.array(new_image_filtered.values))
        i += 1
        
    is_done = True

if get_image_from_vid:
    vid.release()

cv2.destroyAllWindows()

end = time()

print(end - start)