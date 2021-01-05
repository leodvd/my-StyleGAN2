import cv2
import numpy as np
import os
from PIL import Image
from torchvision.transforms import CenterCrop, Resize, FiveCrop, RandomCrop, TenCrop

# PRECROP = 1024
RES = 512

dataset_root = '/home/leo/data_dir/generic_cracks_dataset'
source_path = os.path.join(dataset_root, f'4channels/20180928_flow_5000_resFULL_png_modified-maps')
out_path = os.path.join(dataset_root, f'4channels/20180928_flow_5000_crop{RES}_png_modified-maps_with-crack')


if not os.path.exists(out_path):
    os.mkdir(out_path)

file_names = os.listdir(source_path)
size = len(file_names)

# print(f'Cropping to {PRECROP}*{PRECROP} and resizing to {RES}*{RES}:')
for i, img_name in enumerate(file_names):
    img = Image.open(os.path.join(source_path, img_name))

    img_center_crop = CenterCrop(RES)(img)
    if np.min(np.array(img_center_crop)[:, :, 3]) < 27:
        print(f'center crop of image {i} has a crack')
        img_center_crop.save(f'{os.path.join(out_path, img_name[:-4])}-with-crack.png')
    else:
        found_a_crack_crop = False
        img_crops = [RandomCrop(RES)(img) for z in range(20)]
        img_crops += FiveCrop(RES)(img)
        img_crops += TenCrop(RES)(img)
        for j, img_crop in enumerate(img_crops):
            if not found_a_crack_crop:
                if np.min(np.array(img_crop)[:, :, 3]) < 27:
                    print(f'crop {j+1} of image {i} has a crack')
                    found_a_crack_crop = True
                    img_crop.save(f'{os.path.join(out_path, img_name[:-4])}-with-crack.png')

    if i % round(size/20) == 0:
        print(f'conversion: {i}/{size} ==> {round(100*i/size, 1)}%')
