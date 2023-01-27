import tqdm
import numpy as np

from skimage.feature import greycomatrix, greycoprops
from multiprocessing import Pool

def glcm_props(patch):
    lf = []
    props = ['dissimilarity', 'contrast', 'homogeneity', 'energy', 'correlation']

    # left nearest neighbor
    glcm = greycomatrix(patch, [3], [0], 256, symmetric=True, normed=True)
    for f in props:
        lf.append( greycoprops(glcm, f)[0,0] )

    # upper nearest neighbor
    glcm = greycomatrix(patch, [3], [np.pi/2], 256, symmetric=True, normed=True)
    for f in props:
        lf.append( greycoprops(glcm, f)[0,0] )
        
    return lf

def patch_gen(img, PAD=4):
    img1 = (img * 255).astype(np.uint8)

#     W = 512
    W, H = img.shape[0], img.shape[1]
    imgx = np.zeros((W+PAD*2, H+PAD*2), dtype=img1.dtype)
    imgx[PAD:W+PAD,PAD:H+PAD] = img1
    imgx[:PAD,  PAD:H+PAD] = img1[PAD:0:-1,:]
    imgx[-PAD:, PAD:H+PAD] = img1[W-1:-PAD-1:-1,:]
    imgx[:, :PAD ] = imgx[:, PAD*2:PAD:-1]
    imgx[:, -PAD:] = imgx[:, H+PAD-1:-PAD*2-1:-1]

    xx, yy = np.meshgrid(np.arange(0, W), np.arange(0, W))
    xx, yy = xx.flatten() + PAD, yy.flatten() + PAD

    for x, y in zip(xx, yy):
        patch = imgx[y-PAD:y+PAD+1, x-PAD:x+PAD+1]
        return patch

def glcm_feature(img, verbose=False):
    
#     W, NF, PAD = 512, 10, 4
    
    W, H, NF, PAD = img.shape[0], img.shape[1], 10, 4

    if img.sum() == 0:
        return np.zeros((W,H,NF), dtype=np.float32)
    
    l = []
    with Pool(10) as pool:
        for p in tqdm.tqdm(pool.map(glcm_props, patch_gen(img, PAD)), total=W*H, disable=not verbose):
            l.append(p)
        
    fimg = np.array(l, dtype=np.float32).reshape(img.shape[0], img.shape[1], -1)
    return fimg

