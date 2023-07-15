import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
import numpy as np


# Image processing functions

# Open single band data (eg. DEM, LULC, etc)
def open_data(img_path):
    with rasterio.open(img_path, 'r') as src:
        img = src.read(1)
        meta = src.meta
    return img, meta

# Open multispectral imagery (eg. Sentinel-2 10 bands, Planetscope 4-band)
def open_multiband(img_path):
    with rasterio.open(img_path, 'r') as src:
        b = src.read(1) #blue
        g = src.read(2) #green
        r = src.read(3) #red
        n = src.read(4) #near infrared
        meta = src.meta
    img = np.dstack((b,g,r,n)) # This is a 4band image in b,g,r,n
    
    return img, meta

# Clip % of data histogram for visualization (For satellite images)
def clip(img, percentile):
    out = np.zeros_like(img.shape[2])
    for i in range(img.shape[2]):
        a = 0 
        b = 255 
        c = np.percentile(img[:,:,i], percentile)
        d = np.percentile(img[:,:,i], 100 - percentile)        
        t = a + (img[:,:,i] - c) * (b - a) / (d - c)    
        t[t<a] = a
        t[t>b] = b
        img[:,:,i] =t
    rgb_out = img.astype(np.uint8)   
    return rgb_out

def minmax(img):
    return (img - np.nanmin(img)) / (np.nanmax(img) + np.nanmin(img))

def standardize(img):
    stdev_img = np.nanstd(img)
    mean_img = np.nanmean(img)
    
    return (img - mean_img) / stdev_img

def plot_img(img, form):
    
    plt.figure(figsize = (12,10))
    
    # Assuming band order : B-G-R-NIR 
    if form == 'rgb':
        plt.imshow(np.dstack((img[:,:,2], img[:,:,1], img[:,:,0])))
                   
    elif form == 'false':
        plt.imshow(np.dstack((img[:,:,3], img[:,:,2], img[:,:,1])))
    
    plt.colorbar()
    
    
def reproj_img(input_img, dst_crs):
#     reproj_img(img,'EPSG:4326')
    with rasterio.open(input_img) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
    
        output_img = input_img.split('.')[0]+'_reproj.tif'
    
        with rasterio.open(output_img, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
                
# dst_crs = 'EPSG:4326'
# output_img = 'bs00_test.tif'

def reproj_from(input_img, transform_img, dst_crs):

    with rasterio.open(transform_img) as img_src:
        img_transform, img_width, img_height = calculate_default_transform(
            img_src.crs, dst_crs, img_src.width, img_src.height, *img_src.bounds)

    with rasterio.open(input_img) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()

        kwargs.update({
            'crs': img_src.crs,
            'transform': img_transform,
            'width': img_width,
            'height': img_height
        })

        output_img = input_img.split('.')[0]+'_reproj.tif'
        with rasterio.open(output_img, 'w', **kwargs) as dst:
#             for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=img_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
    
    
def resize(input_img, transform_img):
    with rasterio.open(transform_img) as transform_src:
        transform_img = transform_src.read(1)
        transform_meta = transform_src.meta.copy()
    
    transform_meta
    
    with rasterio.open(input_img) as src:
        w = src.read(1, window=Window(0, 0, width, height))
        profile = src.profile
        profile['width'] = width
        profile['height'] = height

        # Create output
        result = numpy.full((width, height), dtype=profile['dtype'], fill_value=profile['nodata'])
        output_img = input_img.split('.')[0]+'_resized.tif'
    
    # Write
    with rasterio.open(output_img, 'w', **profile) as dataset:
        dataset.write_band(1, result)
