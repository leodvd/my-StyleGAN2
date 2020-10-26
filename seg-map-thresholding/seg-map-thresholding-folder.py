import cv2
import numpy as np
import os
import glob

source_path = '/Users/leopolddavid/Downloads/10_tiles_images_and_masks_512/'
output_path = '/Users/leopolddavid/Downloads/10_tiles_images_and_final_masks_512/'


def my_func(x):
    if 2 < x < 80:
        return 255
    elif x > 80:
        return 0
    else:
        return x


vect_func = np.vectorize(my_func)


if not os.path.exists(output_path):
    os.mkdir(output_path)

file_names = os.listdir(source_path)
nb = len(file_names)
# print(nb)

counter = 1
for img_name in file_names:

    # print(img_name)
    if img_name[-4:] == '.png' and img_name[-8:] != 'mask.png':
        # print(img_name)
        img = cv2.imread(os.path.join(source_path, img_name), cv2.IMREAD_UNCHANGED)

        mask_name = img_name[:-4] + '-mask.png'
        grey_mask = cv2.imread(os.path.join(source_path, mask_name), cv2.IMREAD_UNCHANGED)
        # modified_map = vect_func(grey_mask)
        modified_map = np.vectorize(my_func)(grey_mask)

        output_map_path = os.path.join(output_path, f'{counter}-mask.png')
        cv2.imwrite(filename=output_map_path, img=modified_map)

        output_img_path = os.path.join(output_path, f'{counter}.png')
        cv2.imwrite(filename=output_img_path, img=img)

        counter += 1
