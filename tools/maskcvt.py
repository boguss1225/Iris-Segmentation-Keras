from scipy.spatial import cKDTree
from PIL import Image
import cv2
import numpy
import glob
import sys
import numpy as np
import os
np.set_printoptions(threshold=sys.maxsize)

path = '/home/heemoon/Desktop/0_DATABASE/3_IRIS/cow/labels_roi' # Source Folder
dstpath = '/home/heemoon/Desktop/0_DATABASE/3_IRIS/cow/iii_format' # Destination Folder

#color mapping [R,G,B]
# VOC_COLORMAP = [[0, 0, 0], [51, 221, 255], [255, 0, 204]] #Salmon
# VOC_COLORMAP = [ [0, 0, 0], 
#             [201, 133, 56], # "Echinoderms "
#             [208, 192, 74], # "Molluscs "
#             [225, 100, 204],# "Seaspiders "
#             [128, 72, 199], # "Worms "
#             [91, 193, 127], # "Crustacea "
#             [168, 228, 11], # "Fish "
#             [196, 234, 90], # "Jellies "
#             [228, 184, 126],# "NoID "
#             [252, 97, 214], # "Tube "
#             [99, 31, 21]]   # "sessile fauna "
VOC_COLORMAP = [[0, 0, 0], [255, 255, 255]] #diatom

#@save
# VOC_CLASSES = ['background', 'type4', 'type5'] #Salmon

#@convert & save
def convert_to_iii(file_dir):
    """Build an RGB color to label mapping for segmentation."""
    im=cv2.imread(file_dir)

    # When you want to check color
    print("color list")
    all_rgb_codes = im.reshape(-1, im.shape[-1])
    print(np.unique(all_rgb_codes, axis=0))
    
    # If the image color is noisy
    all_rgb_codes = im.reshape(-1, im.shape[-1])
    if len(VOC_COLORMAP)<len(np.unique(all_rgb_codes, axis=0)):
        cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        colors = np.array(VOC_COLORMAP)
        im = colors[cKDTree(colors).query(im,k=1)[1]]
        all_rgb_codes = im.reshape(-1, im.shape[-1])
        print(np.unique(all_rgb_codes, axis=0))
    
    #convert color here
    for i, colormap in enumerate(VOC_COLORMAP):
        im[np.where((im == [colormap[2],colormap[1],colormap[0]]).all(axis = 2))] = [i,i,i]
    filename = os.path.split(file_dir)[-1]
    cv2.imwrite((os.path.join(dstpath,filename)),im)

#@iterate folder
def voc_label_indices(folder_dir):
    explore_path = folder_dir + "/*.png"
    path_list = glob.glob(explore_path)
    print("How many files?:",len(path_list))
    for file_dir in path_list:
        print(file_dir)
        convert_to_iii(file_dir)

copy_bool = input("Copy target directory before excute (it will overwirte). Excute? (Y/N):")

if copy_bool=='y' or copy_bool=='Y':
    print("Start conversion")
    voc_label_indices(path)
    print("Done")
else : 
    print("See you")
