import cv2
import numpy as np
from skimage.transform import rescale
import os
import glob
from PIL import Image
from torchvision.transforms import CenterCrop, Resize

RES = 512

# source_path = '/home/leo/data_dir/generic_cracks_dataset/raw/0828_flow'
# out_path = f'/home/leo/data_dir/generic_cracks_dataset/cropped_resized/0828_flow_res{RES}'
# source_path = '/home/leo/data_dir/generic_cracks_dataset/raw/20180928_flow_5000'
# out_path = f'/home/leo/data_dir/generic_cracks_dataset/cropped_resized/20180928_flow_5000_res{RES}'
source_path = '/home/leo/data_dir/generic_cracks_dataset/masks255/0828_flow'
out_path = f'/home/leo/data_dir/generic_cracks_dataset/masks255_cropped_resized/0828_flow_res{RES}'

if not os.path.exists(out_path):
    os.mkdir(out_path)

file_names = os.listdir(source_path)
size = len(file_names)

for i, img_name in enumerate(file_names):
    img = Image.open(f'{source_path}/{img_name}')
    img = CenterCrop(1024)(img)
    img = Resize(RES)(img)
    img.save(f'{out_path}/{img_name}')
    if i % round(size/20) == 0:
        print(f'conversion: {i}/{size} ==> {round(100*i/size, 1)}%')
