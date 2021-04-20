# Author: Liam Coulter, from code written by @author: avanetten on Github
# Description: Script to read in images and corresponding geojson files from
#       a specified raw data folder, and save the mask .tif files and processed
#       images in a processed data folder
# --------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
import numpy as np
import shutil
import glob
import sys
import os

# EDIT THESE PATHS
# 1) folder with images and geojson files (in separate sub-folders)
spacenet_data_dir = '/Users/lcoulter/Documents/EE_5561_Project_2/raw_data/' 
# 2) folder to save building masks & processed images (in separate sub-folders)
spacenet_save_dir = '/Users/lcoulter/Documents/EE_5561_Project_2/processed_data/' 
# 3) current directory
spacenet_explore_dir = '/Users/lcoulter/Documents/EE_5561_Project_2/'

# number of images and starting point
N_ims = 6940 # this constrols how many images we input at one time
im_start = 0 # where the image indices start

# import packages
sys.path.extend([spacenet_explore_dir])
from image_utilities import geojson_to_pixel_arr, create_dist_map, \
                            create_building_mask, plot_truth_coords, \
                            plot_building_mask, plot_dist_transform, \
                            plot_all_transforms

# import spacenet utilities
path_to_spacenet_utils = '/EE_5561_Project_2/utilities/'
sys.path.extend([path_to_spacenet_utils])
from spacenetutilities import geoTools as gT

def main():    

    imDir = os.path.join(spacenet_data_dir, 'SN1_buildings_train_AOI_1_Rio_3band/')
    vecDir = os.path.join(spacenet_data_dir, 'SN1_buildings_train_AOI_1_Rio_geojson/')
    imDir_out = os.path.join(spacenet_save_dir, 'processed_3band/') # save processed images once done

    pos_val = 1 # positive value for image masks

    ########################
    # Create directory(ies)
    maskDir = os.path.join(spacenet_save_dir, 'building_mask')

    for p in [maskDir, imDir_out]:
        if not os.path.exists(p):
            os.mkdir(p)

    # get input images and copy to working directory
    rasterList = glob.glob(os.path.join(imDir, '*.tif'))[im_start:im_start+N_ims]   
    for im_tmp in rasterList:
        shutil.copy(im_tmp, imDir_out)

    pixel_coords_list = []
    for i,rasterSrc in enumerate(rasterList):
        
        print((i, "Evaluating", rasterSrc))

        input_image = plt.imread(rasterSrc) # cv2.imread(rasterSrc, 1)
        
         # get name root
        name_root0 = rasterSrc.split('/')[-1].split('.')[0]
        # remove 3band or 8band prefix
        name_root = name_root0[6:]
        vectorSrc = os.path.join(vecDir, 'Geo_' + name_root + '.geojson')
        maskSrc = os.path.join(maskDir, name_root0 + '.tif')

        # create & save mask
        outfile = os.path.join(maskDir, name_root0 + '.tif')
        create_building_mask(rasterSrc, vectorSrc, npDistFileName=outfile, 
                                                  burn_values=pos_val)

if __name__ == '__main__':
    main() 