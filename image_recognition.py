from perceptron import Multilayer_Perceptron, Activation_Functions
from time import time
import cv2

nns = []

time_to_read = 0

for i in range(15):
    s = time()
    nns.append(Multilayer_Perceptron([], 0, [], 0, 0, f'./Multilayer_Perceptrons/{i}.txt', True))
    e = time()
    print(f'/\-{i}\n--|{e-s}s\n--|{int((e-s)//60)}m {int((e-s)%60)}s\n')
    time_to_read += e-s
print(f'/\-/\ \n--|{time_to_read}s\n--|{int(time_to_read//60)}m {int(time_to_read%60)}s\n')
is_done = False

results = []

s = time()
results.append(nns[i].calculate([item for sublist in current_image.values for item in sublist]))
e = time()
print(f'/\-Evaluate {i}\n--|{e-s}s\n{int((e-s)//60)}m {int((e-s)%60)}s\n')
print(results[-1])

print(results,'\n')
print(Activation_Functions.SOFTMAX[0](Activation_Functions, [sum([result[0] for result in results]), sum([result[1] for result in results])]))

