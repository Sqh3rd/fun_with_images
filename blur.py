from matrix import Matrix, kernel_multiplicate
import cv2
import numpy as np

vid = cv2.VideoCapture(0)

check, frame = vid.read()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

cv2.imwrite('original_image.jpg', frame)

cv2.imwrite('new_image.jpg', np.array(kernel_multiplicate(first_matrix=Matrix(values=[[1,-1,1],[-1,1,-1],[1,-1,1]]), second_matrix=Matrix(values=frame), crop_to_val=255).values))
