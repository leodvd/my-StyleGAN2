import cv2
import numpy as np
import os
from PIL import Image
from torchvision.transforms import CenterCrop, Resize, FiveCrop, RandomCrop, TenCrop

RES = 256
percentage_of_crack_images = 80

dataset_root = '/home/leo/data_dir/generic_cracks_dataset'
source_path_no_cracks = os.path.join(dataset_root, f'4channels/20180928_flow_5000_crop512_png_modified-maps_no-cracks')
source_path_with_cracks = os.path.join(dataset_root, f'4channels/20180928_flow_5000_crop512_png_modified-maps_with-cracks')
out_path = os.path.join(dataset_root, f'4channels/20180928_flow_5000_crop512_resize{RES}_png_modified-maps_{percentage_of_crack_images}-percent-cracks')
# out_path = os.path.join(dataset_root, f'4channels/20180928_flow_5000_crop512_png_modified-maps_{percentage_of_crack_images}-percent-cracks')

if not os.path.exists(out_path):
    os.mkdir(out_path)

file_names_no_cracks = os.listdir(source_path_no_cracks)
file_names_with_cracks = os.listdir(source_path_with_cracks)
size = len(file_names_with_cracks)
nb_of_wanted_no_crack_images = round(size*(100-percentage_of_crack_images)/100)

for i, img_name in enumerate(file_names_no_cracks):
    if i < nb_of_wanted_no_crack_images:
        img = Image.open(os.path.join(source_path_no_cracks, img_name))
        img = Resize(RES)(img)
        img.save(os.path.join(out_path, img_name))

        if i % round(size/20) == 0:
            print(f'no crack images conversion: {i}/{nb_of_wanted_no_crack_images} ==> {round(100*i/nb_of_wanted_no_crack_images, 1)}%')

for i, img_name in enumerate(file_names_with_cracks):
    if not os.path.isfile(os.path.join(out_path, f'{img_name[:-14]}no-crack.png')):
        img = Image.open(os.path.join(source_path_with_cracks, img_name))
        img = Resize(RES)(img)
        img.save(os.path.join(out_path, img_name))

    if i % round(size/20) == 0:
        print(f'crack images conversion/checking: {i}/{size} ==> {round(100*i/size, 1)}%')
