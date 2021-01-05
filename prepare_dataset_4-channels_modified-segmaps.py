import os
from PIL import Image
import cv2
import glob
import numpy as np


RES = 512  # 128, 256, 512
dataset_root = '/home/leo/data_dir/generic_cracks_dataset'
source_path = os.path.join(dataset_root, f'cropped_resized/0828_flow_res{RES}')
mask_path = os.path.join(dataset_root, f'masks255_cropped_resized/0828_flow_res{RES}')
output_path = os.path.join(dataset_root, f'4channels/0828_flow_res{RES}_png_modified-maps')

file_names = os.listdir(source_path)
size = len(file_names)

if size > 0:
    if not os.path.exists(output_path):
        os.mkdir(output_path)


for i, img_name in enumerate(file_names):

    # read original image
    original = cv2.imread(os.path.join(source_path, img_name), cv2.IMREAD_UNCHANGED)

    # get dimensions for resizing mask
    height, width, channels = original.shape

    # read alpha image
    mask_name = f'{img_name[:-4]}-mask.png'
    alpha = cv2.imread(os.path.join(mask_path, mask_name))

    # resize alpha image to match original
    alpha_resized = cv2.resize(alpha, (height, width))

    # split alpha_resized into individual channels
    channels = cv2.split(alpha_resized)

    # apply to 4th channel of original
    image_4chan = np.zeros(shape=(height, width, 4))
    image_4chan[:, :, 0] = original[:, :, 0]
    image_4chan[:, :, 1] = original[:, :, 1]
    image_4chan[:, :, 2] = original[:, :, 2]
    image_4chan[:, :, 3] = channels[0]/(-2.5) + 127

    # write new image file with alpha channel
    output_name = f'{img_name[:-4]}.png'
    cv2.imwrite(os.path.join(output_path, output_name), image_4chan)

    if i % round(size/20) == 0:
        print(f'conversion: {i}/{size} ==> {round(100*i/size, 1)}%')
