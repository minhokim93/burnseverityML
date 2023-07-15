import numpy as np
import pandas as pd
from utils import *
import os,glob

# Input bands and images
def input_data(method=None, basepath=None, patch_size=None, i=None):
    
    '''
    Sampling Method # 1 : Random sampling based on wildfire size
    - Split burn severity AOIs in terms of wildfire size (to avoid bias in weightings)
    - Sample based on relative pixel count

    Sampling Method # 2 : Fixed random sampling sets 
    - Sample fixed number for each burn severity class
    * Same number of samples for all AOIs and across burn severity classes

    Sampling Method # 3 : Sampling by superpixel
    - SLIC
    - Grouby SLIC and random sample

    Sampling Method # 4 : Patch samples 
    '''

    img_train, dem_train, slope_train, lulc_train, aoi_train = paths(basepath)

    # Open input datasets for each AOI
    img_t, meta_img = open_multiband(img_train[i])
    dem_t, meta_dem = open_data(dem_train[i])
    slope_t, meta_slope = open_data(slope_train[i])
    lulc_t, meta_lulc = open_data(lulc_train[i])

    ndvi_t = (img_t[:,:,3]/65535 - img_t[:,:,2]/65535) / (img_t[:,:,3]/65535 + img_t[:,:,2]/65535)
    savi=((img_t[:,:,3]/65535 - img_t[:,:,2]/65535)/(img_t[:,:,3]/65535 +img_t[:,:,2]/65535 +0.5))*(1+0.5)
    
    ndvi_t[ndvi_t < 0.2] = 0
    
    lulc_t[lulc_t == 10] = 1
    lulc_t[lulc_t == 20] = 2
    lulc_t[lulc_t == 30] = 3
    lulc_t[lulc_t == 40] = 4 
    lulc_t[lulc_t == 95]= 5
    lulc_t[lulc_t == 50]= 0
    lulc_t[lulc_t == 60]= 0
    lulc_t[lulc_t == 70]= 0
    lulc_t[lulc_t == 80]= 0
    lulc_t[lulc_t == 90]= 0
    lulc_t[lulc_t == 100]= 0

    x, y = np.gradient(dem_t)
    aspect = np.arctan2(-x, y)
    del x,y

    img_stack = np.dstack((minmax(img_t), 
                           minmax(ndvi_t), 
                           minmax(savi), 
                           minmax(dem_t), 
                           minmax(slope_t), 
                           aspect, 
                           lulc_t))   
    
    
    
    if method == 1:
        return img_stack, meta_img
    
    if method == 4:
        # Zero-padding for input datasets (for patch-based sampling)
        pad_width = int(patch_size/2), int(patch_size/2)
        img_padded = np.pad(img_stack, pad_width=[pad_width,pad_width,(0,0)], mode='constant')
        return img_padded
        

def input_data_patches(method=None, basepath=None, patch_size=None, i=None):
    
    img_train, dem_train, slope_train, lulc_train, aoi_train = paths(basepath)


    # Open input datasets for each AOI
    img_t, meta_img = open_multiband(img_train[i])
    dem_t, meta_dem = open_data(dem_train[i])
    slope_t, meta_slope = open_data(slope_train[i])

    ndvi_t = (img_t[:,:,3]/65535 - img_t[:,:,2]/65535) / (img_t[:,:,3]/65535 + img_t[:,:,2]/65535)
    savi=((img_t[:,:,3]/65535 - img_t[:,:,2]/65535)/(img_t[:,:,3]/65535 +img_t[:,:,2]/65535 +0.5))*(1+0.5)
    
    ndvi_t[ndvi_t < 0.2] = 0

    # Blue, Green, Red, NIR, NDVI, SAVI, DEM, Slope
    img_stack = np.dstack((minmax(img_t), 
                           minmax(ndvi_t), 
                           minmax(savi), 
                           minmax(dem_t), 
                           minmax(slope_t), 
                           ))    
    
    if method == 1:
        return img_stack, meta_img
    
    if method == 4:
        # Zero-padding for input datasets (for patch-based sampling)
        pad_width = int(patch_size/2), int(patch_size/2)
        img_padded = np.pad(img_stack, pad_width=[pad_width,pad_width,(0,0)], mode='constant')
        return img_padded
     

        
# Read all pixels for each burn severity class
def read_aoi(aoi_img):
    
    aoi_t, aoi_meta = open_data(aoi_img)
    aoi_t[aoi_t > 4] = 0 # Set nodata values to 0 or NaN

    aoi_low = np.copy(aoi_t)
    aoi_low[aoi_low > 1] = 0

    aoi_mlow = np.copy(aoi_t)
    aoi_mlow[aoi_mlow != 2] = 0

    aoi_mhigh = np.copy(aoi_t)
    aoi_mhigh[aoi_mhigh != 3] = 0

    aoi_high = np.copy(aoi_t)
    aoi_high[aoi_high != 4] = 0
    
    return aoi_t, aoi_low, aoi_mlow, aoi_mhigh, aoi_high


# Method 1 : Random Sampling 
def aoi_sampling(img_stack, aoi_class, sev_num, sample_count, random_state, img_stack_meta):

    # Set mask based on burn severity class: Low = 1 / Med-Low = 2 / Med-High = 3 / High = 4
    aoi_mask = np.where(aoi_class == sev_num)

    # Create AOI df to input random sampling data
    aoi_df = pd.DataFrame()
    aoi_df['c1'] = aoi_mask[0]
    aoi_df['c2'] = aoi_mask[1]
    sampled_aois = aoi_df.sample(n=sample_count, replace=True, random_state=random_state) # Randomly sampled AOI coordinates
#     sampled_aois.reset_index(inplace=False, drop=True)
    
    # Empty placers for sampled features
    sampled_aois['features'] = np.zeros(sampled_aois.shape[0])
    sampled_features = []
    lons, lats = [],[]
    
    # Extract features (as points) from stacked images using randomly sampled points
    for j in sampled_aois.index:
        img_stack_train = img_stack[aoi_df['c1'][j], aoi_df['c2'][j],:] 
        sampled_features.append(img_stack_train)
        
        lon,lat  = rasterio.transform.xy(img_stack_meta['transform'], aoi_df['c1'][j], aoi_df['c2'][j])
        lons.append(lon)
        lats.append(lat)
        
    # Resulting samples as a dataframe with features in a list format
    sampled_aois['longitude'] = lons
    sampled_aois['latitude'] = lats
    sampled_aois['features'] = sampled_features 
    
    return sampled_aois



# Method 2 : Fixed Sampling 
def fixed_sampling(img_stack, aoi_class, sev_num, sample_count, random_state):

    # Set mask based on burn severity class: Low = 1 / Med-Low = 2 / Med-High = 3 / High = 4
    aoi_mask = np.where(aoi_class == sev_num)

    # Create AOI df to input random sampling data
    aoi_df = pd.DataFrame()
    aoi_df['c1'] = aoi_mask[0]
    aoi_df['c2'] = aoi_mask[1]
    
    sampled_aois = aoi_df.sample(n=sample_count, replace=True, random_state=random_state) # Randomly sampled AOI coordinates
#     sampled_aois.reset_index(inplace=False, drop=True)
    
    # Empty placers for sampled features
    sampled_aois['features'] = np.zeros(sampled_aois.shape[0])
    sampled_features = []

    # Extract features (as points) from stacked images using randomly sampled points
    for j in sampled_aois.index:
        img_stack_train = img_stack[aoi_df['c1'][j], aoi_df['c2'][j],:] 
        sampled_features.append(img_stack_train)

    # Resulting samples as a dataframe with features in a list format
    sampled_aois['features'] = sampled_features 
    
    return sampled_aois

# Method 3 : Time-based random sampling
def time_sampling(img_stack, aoi_class, sev_num, sample_count, random_state, img_stack_meta):

    # Set mask based on burn severity class: Low = 1 / Med-Low = 2 / Med-High = 3 / High = 4
    aoi_mask = np.where(aoi_class == sev_num)

    # Create AOI df to input random sampling data
    aoi_df = pd.DataFrame()
    aoi_df['c1'] = aoi_mask[0]
    aoi_df['c2'] = aoi_mask[1]
    sampled_aois = aoi_df.sample(n=sample_count, replace=True, random_state=random_state) # Randomly sampled AOI coordinates
#     sampled_aois.reset_index(inplace=False, drop=True)
    
    # Empty placers for sampled features
    sampled_aois['features'] = np.zeros(sampled_aois.shape[0])
    sampled_features = []
    lons, lats = [],[]
    
    # Extract features (as points) from stacked images using randomly sampled points
    for j in sampled_aois.index:
        img_stack_train = img_stack[aoi_df['c1'][j], aoi_df['c2'][j],:] 
        sampled_features.append(img_stack_train)
        
        lon,lat  = rasterio.transform.xy(img_stack_meta['transform'], aoi_df['c1'][j], aoi_df['c2'][j])
        lons.append(lon)
        lats.append(lat)
        
    # Resulting samples as a dataframe with features in a list format
    sampled_aois['longitude'] = lons
    sampled_aois['latitude'] = lats
    sampled_aois['features'] = sampled_features 
     
    return sampled_aois

# Method 4 : Patch-based random sampling
def mean_sampling(img_stack, img_padded, aoi_class, sev_num, sample_count, patch_size, random_state):

    # Set mask based on burn severity class: Low = 1 / Med-Low = 2 / Med-High = 3 / High = 4
    aoi_mask = np.where(aoi_class == sev_num)

    # Create AOI df to input random sampling data
    aoi_df = pd.DataFrame()
    aoi_df['c1'] = aoi_mask[0]
    aoi_df['c2'] = aoi_mask[1]
    sampled_aois = aoi_df.sample(n=sample_count, replace=True, random_state=random_state) # Randomly sampled AOI coordinates

    # Empty placers for sampled features
    sampled_aois['features'] = np.zeros((sampled_aois.shape[0]))
    sampled_features, sampled_features1, sampled_features2 = [],[],[]

    # Extract features (as patches) from stacked images using randomly sampled points
    for j in sampled_aois.index:
        img_stack_train = img_stack[aoi_df['c1'][j], aoi_df['c2'][j],:] 
        sampled_features1.append(img_stack_train)
        
        centroid = aoi_df['c1'][j], aoi_df['c2'][j]
        low_b = int(centroid[0] - patch_size/2), int(centroid[0] + patch_size/2) 
        high_b = int(centroid[1] - patch_size/2), int(centroid[1] + patch_size/2)

        patchs = img_padded[low_b[0]:low_b[1], high_b[0]:high_b[1]]
        mean_patch = [np.nanmean(patchs[:,:,band]) for band in range(patchs.shape[2])]
        sampled_features2.append(mean_patch)
        
        sampled_features = list(np.hstack((sampled_features1,sampled_features2)))

    # Resulting samples as a dataframe with features in a list format
    sampled_aois['features'] = sampled_features
   
    return sampled_aois


# Method 4 : Patch-based random sampling
def patch_sampling(img_padded, aoi_class, sev_num, sample_count, patch_size, random_state):

    # Set mask based on burn severity class: Low = 1 / Med-Low = 2 / Med-High = 3 / High = 4
    aoi_mask = np.where(aoi_class == sev_num)

    # Create AOI df to input random sampling data
    aoi_df = pd.DataFrame()
    aoi_df['c1'] = aoi_mask[0]
    aoi_df['c2'] = aoi_mask[1]
    sampled_aois = aoi_df.sample(n=sample_count, replace=True, random_state=random_state) # Randomly sampled AOI coordinates

    # Empty placers for sampled features
    sampled_aois['features'] = np.zeros(sampled_aois.shape[0])
    sampled_features = []

    # Extract features (as patches) from stacked images using randomly sampled points
    for j in sampled_aois.index:

        centroid = aoi_df['c1'][j], aoi_df['c2'][j]
        low_b = int(centroid[0] - patch_size/2), int(centroid[0] + patch_size/2) 
        high_b = int(centroid[1] - patch_size/2), int(centroid[1] + patch_size/2)
        
#         print("Coordinates: ", low_b, high_b)
#         print("Patch size: ", low_b[1]-low_b[0], high_b[1]-high_b[0])

        patchs = img_padded[low_b[0]:low_b[1], high_b[0]:high_b[1]]
        sampled_features.append(patchs)

    # Resulting samples as a dataframe with features in a list format
    sampled_aois['features'] = sampled_features
   
    return sampled_aois


def paths(base_path):
    
    os.chdir(base_path)

    img_train = sorted(glob.glob('s2*.tif'))
    dem_train = sorted(glob.glob('dem*.tif'))
    slope_train = sorted(glob.glob('slope*.tif'))
    aoi_train = sorted(glob.glob('fire*.tif'))
    lulc_train = sorted(glob.glob('lulc*.tif'))

    return img_train, dem_train, slope_train, lulc_train, aoi_train