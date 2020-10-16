# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:38:03 2020

@author: avh492
"""

import numpy as np
import glob
from PIL import Image
import json
import os
import cv2
import tensorflow 
from tensorflow.keras.utils import to_categorical
import post_utils as pu
from matplotlib import pyplot as plt
from skimage.transform import rotate

tif_loc         ='train'
im_size         =(640,560)   #remember that this is wierd and backwards
npy_loc         = 'numpys'     

def im_resize(X, im_size, intp = 'area'):
    '''
    Parameters
    ----------
    X : image 
    im_size : new image dimensions listed (width , height)
    intp : interpolation algorithm, inter_area preserves more information, but nearest neighbor is needed to 
            preserve specific values in y
    Returns
    -------
    a numpy array of size (im_size), type uint8
    '''
    if intp == 'area':
        a = cv2.resize(np.float32(X), im_size, interpolation = cv2.INTER_AREA) 
    elif intp == 'nn':
        a = cv2.resize(np.float32(X), im_size, interpolation = cv2.INTER_NEAREST) 
    return np.array(a).astype(np.uint8)

def augmentation(img, im_size, aug_num, angle, X):
    '''
    creates additional training images by flipping and rotating the original images
    rotate(mode = reflect): mirrors the original data to fill gaps caused by rotating and fitting to im_size
    rotate(preserve_range & order): ensure that intensity values are consistant after rotation
    '''

    if X ==False:
        dims = (aug_num + 1, img.shape[0], img.shape[1])
        aug = np.zeros(dims)
        aug[0,:,:] =  img
        aug[1,:,:] = np.flip(img,0)
        aug[2,:,:] = np.flip(img,1)
        aug[3,:,:] = np.flip(np.flip(img,0),1)
        aug[4,:,:] = rotate(img ,angle, mode='reflect', preserve_range=True, order=0)
        aug[5,:,:] = rotate(aug[1,:,:] ,angle, mode='reflect', preserve_range=True, order=0)
        aug[6,:,:] = rotate(aug[2,:,:] ,angle, mode='reflect', preserve_range=True, order=0)
        aug[7,:,:] = rotate(aug[3,:,:] ,angle, mode='reflect', preserve_range=True, order=0)
    if X ==True:
        dims = (aug_num + 1, img.shape[0], img.shape[1], img.shape[2])
        aug = np.zeros(dims)
        aug[0,:,:,:] =  img
        aug[1,:,:,:] = np.flip(img,0)
        aug[2,:,:,:] = np.flip(img,1)
        aug[3,:,:,:] = np.flip(np.flip(img,0),1)
        aug[4,:,:,:] = rotate(img ,angle, mode='reflect', preserve_range=True, order=0)
        aug[5,:,:,:] = rotate(aug[1,:,:,:] ,angle, mode='reflect', preserve_range=True, order=0)
        aug[6,:,:,:] = rotate(aug[2,:,:,:] ,angle, mode='reflect', preserve_range=True, order=0)
        aug[7,:,:,:] = rotate(aug[3,:,:,:] ,angle, mode='reflect', preserve_range=True, order=0)
    return aug
        
def create_dataset(tif_loc, npy_loc, im_size):
    '''
    - Generates the training dataset of .npy files from a series of tiff images
    - each channel is normalized to the mean intensity
    - in addition to x and y, also generates cell (from red channel) and nuclei (blue channel) binary masks
    - in a loop to avoid maxing out memory for large datasets
    '''
    
    all_files = glob.glob(f'../Images/X_{tif_loc}/*_B*')
    X_count         = 0
    y_count         = 0
    mask_count      = 0
    nuclei_count    = 0
    for num in range( len(all_files) ):
        print(f'reading_{num:04}')        
        X_red = cv2.imread(glob.glob(f'../Images/X_{tif_loc}/*_{num:04}_R*.tif')[0])
        X_green = cv2.imread(glob.glob(f'../Images/X_{tif_loc}/*_{num:04}_G*.tif')[0])
        X_blue = cv2.imread(glob.glob(f'../Images/X_{tif_loc}/*_{num:04}_B*.tif')[0])
        y_img = cv2.imread(glob.glob(f'../Images/Y_{tif_loc}/*_{num:04}*.tif')[0])
        
        X_R     = np.array(im_resize(X_red[:,:,0], im_size), dtype='float64') 
        X_G     = np.array(im_resize(X_green[:,:,0], im_size), dtype='float64') 
        X_B     = np.array(im_resize(X_blue[:,:,0], im_size), dtype='float64') 
        y       = np.array(im_resize(y_img[:,:,0], im_size, intp = 'nn' ))
        
        y[y==128] = 1
        y[y==255] = 2
        y[y>2   ] = 0   #sanity check
        if len( np.unique(y) ) > 3:
            raise NameError('The output has more than 3 values')
            
        mean_dic = json.load(open('../numpys/mean_dic_train.json'))
       
        X_R -= mean_dic['mean_R']
        X_G -= mean_dic['mean_G']
        X_B -= mean_dic['mean_B']
        
        X = np.concatenate(( np.expand_dims(X_R,2), np.expand_dims(X_G,2), np.expand_dims(X_B,2) ), 2)
        mask = pu.get_mask(X)
        nuclei = pu.get_nuclei(X)
        
        X_aug       = augmentation(X, im_size, 7, 45, X=True).astype(np.float16)
        y_aug       = augmentation(y, im_size, 7, 45, X=False)
        y_aug       = to_categorical(y_aug).astype(np.uint8)   
        mask_aug    = augmentation(mask, im_size, 7, 45, X=False).astype(np.float16)
        nuclei_aug  = augmentation(nuclei, im_size, 7, 45, X=False).astype(np.float16)

        for i in range(X_aug.shape[0]):
            np.save(f'../{npy_loc}/X_train/{X_count:04}', X_aug[i,:,:,:])
            X_count += 1
            np.save(f'../{npy_loc}/y_train/{y_count:04}', y_aug[i,:,:])
            y_count += 1
            np.save(f'../{npy_loc}/Binary_R_train/{mask_count:04}', mask_aug[i,:,:])
            mask_count += 1
            np.save(f'../{npy_loc}/Binary_B_train/{nuclei_count:04}', nuclei_aug[i,:,:])
            nuclei_count += 1
        

create_dataset(tif_loc, npy_loc, im_size)       
        
        
        
        
        
        
        