'''
Deep learning based burn severity prediction using remote sensing data
: Utility functions
- Author: Minho Kim (2023)
'''

# Load libraries
import numpy as np
import rasterio
from patchify import patchify, unpatchify

import torch
import torch.nn.functional as F

# One hot encoding
def one_hot_encode(labels, num_classes):
    labels = torch.tensor(labels, dtype=torch.int64)  # Convert labels to a PyTorch tensor with int64 dtype    
    one_hot = F.one_hot(labels, num_classes)
    # .permute(2,1,0)
    
    return one_hot

# Stack images
def open_data(s2_path, dem_path, slope_path, lulc_path, burn_severity_path):

    img_stack = []

    b,g,r,n = rasterio.open(s2_path, 'r').read([1,2,3,4])
    dem = rasterio.open(dem_path, 'r').read([1])[0]
    slope = rasterio.open(slope_path, 'r').read([1])[0]
    ndvi = (n - r) / (n + r + 1e-8) 
    savi = ( (n - r) / (n + r + 0.5) ) * 1.5
    ndwi = (g - n) / (g + n + 1e-8)

    lulc_img = (rasterio.open(lulc_path, 'r').read([1])[0] / 10).astype(int)
    # lulc_img[lulc_img==10] = 1
    # lulc_img[lulc_img==20] = 2
    # lulc_img[lulc_img==30] = 3
    # lulc_img[lulc_img==40] = 4
    # lulc_img[lulc_img==50] = 5
    # lulc_img[lulc_img==60] = 6
    # lulc_img[lulc_img==70] = 7
    # lulc_img[lulc_img==80] = 8
    # lulc_img[lulc_img==90] = 9
    # lulc_img[lulc_img==100] = 100
    
    # lulc = one_hot_encode(lulc_img, 11)
    # lulc = lulc[:, 1:,:] # Index = 0 acts as dummy

    img = np.dstack((b,g,r,n,dem,slope,ndvi,savi,ndwi, lulc_img))
    # img = np.dstack((b,g,r))
    lbl = rasterio.open(burn_severity_path, 'r').read([1])[0]

    return img, lbl

# Min-max normalization (Preprocessing)
def minmax(img):
    a = ( img - np.nanmin(img) ) / ( np.nanmin(img) + np.nanmax(img) )
    return a      

def minmax_bands(image):
    rescaled = [minmax(image[:,:,:,i]) for i in range(image.shape[-1])]
    stack = np.stack(rescaled) # [bands, x, y, batches]
    stack = np.rollaxis(stack, 0, 4) # [batches, x, y, bands]

    return stack

# Prepare patches
def prep_data(image, label, patch_size, mode, threshold):
    
    instances,instances_labels, indexes = [],[],[]

    size_x = (image.shape[1] // patch_size) * patch_size  # width to the nearest size divisible by patch size
    size_y = (image.shape[0] // patch_size) * patch_size  # height to the nearest size divisible by patch size
    
    # Extract patches from each image, step=patch_size means no overlap
    step=patch_size
    
    if mode == "train":
        n_bands = image.shape[2]

        if n_bands > 1:            
            image = image[0:size_x, 0:size_y,:]
            patch_img = patchify(image, (patch_size, patch_size, n_bands), step=patch_size)
        else:
            image = image[0:size_x, 0:size_y]
            patch_img = patchify(image, (patch_size, patch_size), step=patch_size)            

    lbl = label[0:size_x, 0:size_y]
    patch_lbl = patchify(lbl, (patch_size, patch_size), step=patch_size)

    patch_lbl[patch_lbl>4] = 0 # Omit any values above 4 (V high severity)
    patch_lbl[patch_lbl==-9999] = 0 # Omit any null values
    labels = patch_lbl

    # iterate over patch axis
    i=0

    for j in range(patch_img.shape[0]):
        for k in range(patch_img.shape[1]):

            single_img = patch_img[j, k] # patches are located like a grid. use (j, k) indices to extract single patched image
            single_lbl = labels[j, k] # patches are located like a grid. use (j, k) indices to extract single patched image
            
            lbl = single_lbl
            count, num = np.unique(lbl, return_counts=True)
            
            if num[0]/patch_size**2 < threshold:
            # if len(count) > 1 and 0 in count and num[0]/patch_size**2 < threshold: 

                instances.append(single_img[0])
                instances_labels.append(single_lbl)
            
            i += 1 # Increase counter

    indexes.append(len(instances))

    return instances, instances_labels, indexes