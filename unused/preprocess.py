'''
Deep learning based burn severity prediction using remote sensing data
: Preprocessing (Load data, patchify, mask null values, minmax inputs, prepare train/test/val datasets)
- Author: Minho Kim (2023)
'''

from utils import *

import numpy as np
import torch

from sklearn.model_selection import train_test_split

###
def preprocess(s2_path, dem_path, slope_path, lulc_path, burn_severity_path, patch_size, mode, threshold, _train_fraction, _test_fraction, _seed):

    # Load image stacks with label
    full_stack, indexes, labels_list = [],[],[]

    # Loop through all burn severity samples and patchify
    for i in range(len(burn_severity_path)):

        # Open stacked image and label (for one instance)
        img_stack, label = open_data(s2_path[i], dem_path[i], slope_path[i], lulc_path[i], burn_severity_path[i])

        # Prepare data
        imgs, labels, idxs = prep_data(img_stack,label,patch_size, mode, threshold)
        imgs = np.array(imgs)
        labels = np.array(labels)

        # Expand 1-band images to 4D tensors
        if len(np.array(imgs).shape) < 4: 
            imgs = np.expand_dims(imgs, -1)
        
        full_stack.append(imgs)
        indexes.append(idxs[0])
        labels_list.append(labels)

    # Stack
    imgs = np.concatenate((full_stack),axis=0)
    labels = np.concatenate((labels_list),axis=0)

    # Preprocess inputs
    imgs = imgs.astype(float)
    imgs[imgs==-9999]= np.nan
    nan_mask = np.isnan(imgs)
    imgs[nan_mask] = 0

    # Preprocess labels
    labels = labels.astype(float)
    labels[labels==-9999] = np.nan
    nan_mask = np.isnan(labels)
    labels[nan_mask] = 0

    # Dataset split
    X_test = None
    X_train, X_val, Y_train, Y_val = train_test_split(imgs, labels, test_size=_train_fraction, random_state=_seed)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=_test_fraction, random_state=_seed)

    # Minmax Scaling
    # X_train = minmax_bands(X_train)
    # X_val = minmax_bands(X_val)

    print("TRAINING :", X_train.shape, Y_train.shape)
    print("TESTING :", X_test.shape, Y_test.shape)
    print("VALIDATION :", X_val.shape, Y_val.shape)

    # Normalized weights for class imbalance
    num,count = np.unique(labels, return_counts = True) # _labels for one-hot

    # max_class_count = count[np.argmax(count)]
    weights = sum(count) / count
    weights /= max(weights)
    weights = np.append(0, weights) # normalized

    # Converting training images into torch format
    X_train_res = np.rollaxis(X_train, 3,1)
    X_val_res = np.rollaxis(X_val, 3,1)

    # Reshape and prepare tensors for Train and Validation sets
    train_x = X_train_res.reshape(X_train_res.shape[0], X_train_res.shape[1], X_train_res.shape[2], X_train_res.shape[3])
    train_x  = torch.from_numpy(X_train_res).float()
    val_x = X_val_res.reshape(X_val_res.shape[0], X_val_res.shape[1], X_val_res.shape[2], X_val_res.shape[3])
    val_x  = torch.from_numpy(X_val_res).float()

    # Create Test set
    if X_test is not None:
        # X_test = minmax_bands(X_test)
        X_test_res = np.rollaxis(X_test, 3,1)
        test_x = X_test_res.reshape(X_test_res.shape[0], X_test_res.shape[1], X_test_res.shape[2], X_test_res.shape[3])
        test_x  = torch.from_numpy(X_test_res).float()
        # test_y = Y_test
        test_y = Y_test.astype(int)
        test_y = torch.from_numpy(Y_test)
        print("TESTING :", test_x.shape, test_y.shape)
        del X_test_res

    # Converting the target (Labels) into torch format
    train_y = Y_train.astype(int)
    train_y = torch.from_numpy(Y_train)
    val_y = Y_val.astype(int)
    val_y = torch.from_numpy(Y_val)

    # Shape of training data
    print("TRAINING :", train_x.shape, train_y.shape)
    print("VALIDATION :", val_x.shape, val_y.shape)

    del X_train_res, X_val_res

    return train_x, train_y, val_x, val_y, test_x, test_y, weights, imgs.shape[-1], X_train.shape[-1]
