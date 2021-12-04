import cv2
from time import sleep
from matrix import Matrix, kernel_multiplicate
import numpy as np
from time import time

start = time()

vid = cv2.VideoCapture(0)

kernel = Matrix(
    values=[[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
filters = [
    #100
    #010
    #001
    Matrix(values=[[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]),
    #100
    #100
    #100
    Matrix(values=[[1, -1, -1], [1, -1, -1], [1, -1, -1]]),
    #001
    #001
    #001
    Matrix(values=[[-1, -1, 1], [-1, -1, 1], [-1, -1, 1]]),
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
    #001
    #010
    #001
    Matrix(values=[[-1, -1, 1], [-1, 1, -1], [-1, -1, 1]]),
    #010
    #001
    #010
    Matrix(values=[[-1, 1, -1], [-1, -1, 1], [-1, 1, -1]]),
    #100
    #010
    #100
    Matrix(values=[[1, -1, -1], [-1, 1, -1], [1, -1, -1]]),
    #001
    #010
    #010
    Matrix(values=[[-1, -1, 1], [-1, 1, -1], [-1, 1, -1]]),
    #010
    #100
    #100
    Matrix(values=[[-1, 1, -1], [1, -1, -1], [1, -1, -1]]),
    #100
    #010
    #010
    Matrix(values=[[1, -1, -1], [-1, 1, -1], [-1, 1, -1]]),
    #010
    #001
    #001
    Matrix(values=[[-1, 1, -1], [-1, -1, 1], [-1, -1, 1]]),
    #001
    #001
    #010
    Matrix(values=[[-1, -1, 1], [-1, -1, 1], [-1, 1, -1]]),
    #010
    #010
    #100
    Matrix(values=[[-1, 1, -1], [-1, 1, -1], [1, -1, -1]]),
    #010
    #010
    #001
    Matrix(values=[[-1, 1, -1], [-1, 1, -1], [-1, -1, 1]]),
    #100
    #100
    #010
    Matrix(values=[[1, -1, -1], [1, -1, -1], [-1, 1, -1]]),
    #000
    #001
    #110
    Matrix(values=[[-1, -1, -1], [-1, -1, 1], [1, 1, -1]]),
    #001
    #110
    #000
    Matrix(values=[[-1, -1, 1], [1, 1, -1], [-1, -1, -1]]),
    #000
    #100
    #011
    Matrix(values=[[-1, -1, -1], [1, -1, -1], [-1, 1, 1]]),
    #100
    #011
    #000
    Matrix(values=[[1, -1, -1], [-1, 1, 1], [-1, -1, -1]]),
    #000
    #110
    #001
    Matrix(values=[[-1, -1, -1], [1, 1, -1], [-1, -1, 1]]),
    #110
    #001
    #000
    Matrix(values=[[1, 1, -1], [-1, -1, 1], [-1, -1, -1]]),
    #000
    #011
    #100
    Matrix(values=[[-1, -1, -1], [-1, 1, 1], [1, -1, -1]]),
    #011
    #100
    #000
    Matrix(values=[[-1, 1, 1], [1, -1, -1], [-1, -1, -1]]),
    #111
    #000
    #000
    Matrix(values=[[1, 1, 1], [-1, -1, -1], [-1, -1, -1]]),
    #000
    #000
    #111
    Matrix(values=[[-1, -1, -1], [-1, -1, -1], [1, 1, 1]])
]

is_done = False

while not is_done:
    check, frame = vid.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel_multiplicated_image = kernel_multiplicate(
        first_matrix=kernel,
        second_matrix=Matrix(values=frame),
        stride_length=2,
        crop_to_val=255)

    new_image = np.array(kernel_multiplicated_image.values)
    cv2.imwrite("old_image.jpg", frame)
    cv2.imwrite("new_image.jpg", new_image)

    f = 1 / 255
    new_frame = [[b * f for b in a] for a in frame]

    new_images_filtered = []

    for filter in filters:
        new_image_filtered = kernel_multiplicate(
            first_matrix=filter,
            second_matrix=Matrix(values=new_frame),
            stride_length=1,
            crop_to_val=255,
            get_average=True).values
        new_images_filtered.append(new_image_filtered)

    end_image = []
    for a in range(len(new_images_filtered[0])):
        end_image.append([])
        for b in new_images_filtered[0][a]:
            end_image[a].append(0)
    for image in new_images_filtered:
        for a in range(len(image)):
            for b in range(len(image[a])):
                end_image[a][b] += image[a][b]

    for a in range(len(end_image)):
        for b in range(len(end_image[a])):
            if end_image[a][b] > 255:
                end_image[a][b] = 255
            elif end_image[a][b] < 150:
                end_image[a][b] = 0

    print(end_image)
    cv2.imwrite("new_image_filtered.jpg", np.array(end_image))
    print(frame.shape)
    print(new_image.shape)
    is_done = True

vid.release()

cv2.destroyAllWindows()

end = time()

print(end - start)