import cv2
import numpy as np

original_map_path = '/Users/leopolddavid/Downloads/51-ema-mask.png'
output_map_path = original_map_path[:-4] + '-bw.png'


def my_func(x):
    if 2 < x < 80:
        return 255
    elif x > 80:
        return 0
    else:
        return x


original_map = cv2.imread(original_map_path, cv2.IMREAD_UNCHANGED)
modified_map = np.vectorize(my_func)(original_map)
cv2.imwrite(filename=output_map_path, img=modified_map)