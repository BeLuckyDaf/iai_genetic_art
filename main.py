import numpy as np
import cv2
import random
import math
import time
from numba import jit
import genalg
import sys

if len(sys.argv) < 2:
    print("Please provide the file path as a first argument.")
    exit(1)
filepath = sys.argv[1]
target = cv2.imread(filepath, cv2.IMREAD_COLOR)

t = time.time()

image = genalg.calculate_average_image(target, 64)
result = genalg.calculate_image(image, target, epochs=5, goal=10, mutation_rate=0.03, offset=150, select=5, initial_pop=15)
cv2.imshow("source", image)
cv2.imshow("target", target)
cv2.imshow("result", result)

filepath.replace("\\", "/")
nameonly = filepath.split("/")[-1]
cv2.imwrite("result/source_{}".format(nameonly), image)
cv2.imwrite("result/target_{}".format(nameonly), target)
cv2.imwrite("result/result_{}".format(nameonly), result)

print("TIME ELAPSED: {}".format(time.time() - t))

cv2.waitKey(0)
cv2.destroyAllWindows()
