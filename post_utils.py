# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:32:00 2020

@author: avh492
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import skimage
from skimage import filters
from scipy.ndimage.measurements import label as find_conn_comps
import generalized_otsu as gotsu
from nd2reader import ND2Reader as nd2

def get_nuclei(image):
    '''
    - returns segmented nuclei from blue channel 
    - assumes input image is formatted channel 1/2/3 = R/G/B
    '''
    img = image[:,:,2]
    n , bin = np.histogram(img,bins=255)
    thresh, vals = gotsu.GHT(n)     # breaks if histogram peak is 0 (mode of img = 0)
    if thresh <= np.min(img):
        thresh = skimage.filters.threshold_yen(img)
    tri = img < thresh
    image = 255*np.array(tri).astype(np.uint8)
    image = (image==0).astype(int)
    nuclei,b = find_conn_comps(image)
    for comp in range(b):
        comp_size = np.sum(nuclei==comp)
        if comp_size < 50: # 100 pixels is a good min cutoff for nuclei
            nuclei[nuclei==comp] = 0
        if comp_size > 2000:
            nuclei[nuclei==comp] = 0        
    nuclei[nuclei > 0] = 255
    return nuclei


def get_mask(image):
    '''
    - returns segmented cells from red channel
    -assumes input image is formatted channel 1/2/3 = R/G/B
    - inverts image to remove noise / dust from both background and within cells
    '''
    inv =[]
    img = image[:,:,0]
    n , bin = np.histogram(img,bins=255)
    thresh1 = skimage.filters.threshold_triangle(img)
    tri = img <= thresh1
    image = 255*np.array(tri).astype(np.uint8)
    image = (image==0).astype(int)
    a,b = find_conn_comps(image)
    for comp in range(b):
        comp_size = np.sum(a==comp)
        if comp_size < 200: # 100 pixels is a good min cutoff for cells
            a[a==comp] = 0
    a[a > 0] = 255
    inv = a==0 
    inv = inv*255       
    c,d = find_conn_comps(inv)
    for comp in range(d):
        comp_size = np.sum(c==comp)
        if comp_size < 10:
            inv[c==comp] = 0
            
    filled = inv==0
    mask = filled*255
    
    return mask

def det_activation(prediction,nuclei):
    '''
    - overlays the nuclei mask on the prediction to determine activation state
    - assumes that 1 nuclei per cell, and that all cells have a nucleus 
    - returns the number of cells/nuclei in an image and the average activation of all cells
    '''
    nuclei_mask, num_comps = find_conn_comps(nuclei)
    activation_list = []
    for comp in range(num_comps):
        activation = prediction[nuclei_mask==comp+1]
        activation = np.rint(np.sum(activation) / len(activation))
        activation_list = np.append(activation_list, activation)
    activation_list = activation_list[activation_list !=0]
    avg_activation = np.average(activation_list) - 1  # need to subtract 1 becasue acivation values range from 1-2
    num_cells = len(activation_list)
    return avg_activation, num_cells
    
def format_numpy(all_dirs,im_num, img_loc, image_dims):
    '''
    - coverts image set to numpy arrays 
    - Our images are originally saved as .nd2s (default for Nikon microscope)
    - These images are saved as channel 1/2/3 = B/R/G
    '''

    img = np.array(nd2(all_dirs[im_num]), dtype='float64')
    
    img_npy = np.zeros((image_dims[1], image_dims[0],3))
    img_npy[:,:,0] = cv2.resize(img[1,:,:], image_dims, interpolation = cv2.INTER_AREA)
    img_npy[:,:,1] = cv2.resize(img[2,:,:], image_dims, interpolation = cv2.INTER_AREA)
    img_npy[:,:,2] = cv2.resize(img[0,:,:], image_dims, interpolation = cv2.INTER_AREA)
    
    return img_npy

def compare_cells(nuclei, prediction, Y):
    '''
    - compares results of prediction to y_true for test set
    - returns the number of cells as well as the number correctly predicted
    '''
    
    num_correct = 0
    nuc_mask, num_comps = find_conn_comps(nuclei)
    for comp in range(num_comps):
        predicted   = prediction[nuc_mask == comp+1]
        truth       = Y[nuc_mask == comp+1]
        avg_predicted = np.rint(np.sum(predicted) / len(predicted))
        avg_truth = np.rint(np.sum(truth) / len(truth))
        if avg_predicted == avg_truth:
            num_correct += 1
    return num_correct, num_comps
        
        
    
    
    
    
    
    
    
    
    
    