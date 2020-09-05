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

def get_nuclei(image, data_loc, im_num, image_dims):
    
    img = image[:,:,2]
    n , bin = np.histogram(img,bins=255)
    thresh, vals = gotsu.GHT(n)     # can break if hist peak is 0 (mode of img = 0)
    if thresh ==0:
        thresh = skimage.filters.threshold_yen(img)
    tri = img < thresh
    image = 255*np.array(tri).astype(np.uint8)
    image = (image==0).astype(int)
    a,b = find_conn_comps(image)
    for comp in range(b):
        comp_size = np.sum(a==comp)
        if comp_size < 50: # 100 pixels is a good min cutoff for nuclei
            a[a==comp] = 0
        if comp_size > 2000:
            a[a==comp] = 0 
            
            
    a[a > 0] = 255
    nuclei = a
    
    return nuclei


def get_mask(image, data_loc, im_num, image_dims):
    # assuming that image in a .np and that channels 0/1/2 are R/G/B
    # assuming that image_dims is listed as (w x h) in this case (640,560)
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
        if comp_size < 100: # 100 pixels is a good min cutoff for cells
            a[a==comp] = 0
    a[a > 0] = 255
    inv = a==0 
    inv = inv*255       
    c,d = find_conn_comps(inv)
    for comp in range(d):
        comp_size = np.sum(c==comp)
        if comp_size < 50:
            inv[c==comp] = 0
            
    filled = inv==0
    mask = filled*255
    
    return mask

def det_activation(prediction,nuclei):
    # assumes that each nuclei denotes an individual cell
    nuclei_mask, num_comps = find_conn_comps(nuclei)
    num_cells = num_comps
    activation_list = []
    for comp in range(num_comps):
        activation = prediction[nuclei_mask==comp+1]
        activation = np.rint(np.sum(activation) / len(activation))
        activation_list = np.append(activation_list, activation)
    activation_list[activation_list !=0]
    avg_activation = np.average(activation_list)
    return avg_activation, num_cells
    
        
        
    
    
    
    
    
    
    
    
    
    