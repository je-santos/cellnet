# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:26:32 2020

@author: avh492
"""

'''
Things to adjust:
    - change averages that test images are normalized by (in create numpys code)
    - multiply predicction by mask to get real prediction
    -Fix evaluation script (mismatched dimensions)
'''


import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import skimage
from scipy.ndimage.measurements import label as find_conn_comps
import json
import os
import tensorflow as tf
import cell_utils as cu
import post_utils as pu


model_name          = 'cellnet_today_biggest'
image_dims          = (640,560)         # (w x h) is opposite of convention bc cv2 is dumb
data_loc         = '../numpys/X_test'
all_dirs         =glob.glob('../numpys/X_test/*.npy')       
y_dirs           =glob.glob('../numpys/y_test/*.npy')

save_img        = False         # Turn to True to save an image of each prediction
known_test      = True         # Turn to True to use manually pre-processed Ys as a test
record_metrics  = False


try:
    os.mkdir(f'../evaluation/{model_name}')
    print("Directory " , model_name ,  " Created ")
except FileExistsError:
    print("Directory " , model_name ,  " already exists")
    
'''
load the model
'''

try:
    model = tf.keras.models.load_model( 'models/' + model_name + '/' + model_name + '.h5', 
                                      custom_objects={'iou_loss': cu.iou_loss,
                                                      'iou_0': cu.build_iou_for(label=0),
                                                      'iou_1': cu.build_iou_for(label=1)} ) #loads the model
    print('-'*50)
    print(f'Sucefully loaded {model_name}')    
    print('-'*50)

except:
  print('*'*50)
  print('No checkpoint found')
  print('*'*50)

'''
Appply model to make predictions

- Use get_mask to get a binary mask of the cells based on red channel
- Use get_nuclei to get a binary mask of nuclei based on blue channel
- use det_activation to use the nuclei and predicction mask to determine the
    average activation level of all the cells in the image
- Returns "data_set" which contains the number of cells per image and the average activation
- Used a for loop to do 1 image at a time so it could handle large data sets
    i.e. for actual implementation it will be used on 500+ images
'''
data_set = np.zeros((len(all_dirs),2))
for im_num in range( len(all_dirs) ):
    #for im_num in range(1):
    print(f'reading_image_{im_num:04}')
    img = np.zeros((image_dims[1],image_dims[0],3))
    img[:,:,:] = np.load( data_loc + f'/{im_num:04}.npy' )
    
    mask = pu.get_mask(img, data_loc, im_num, image_dims)
    mask = np.concatenate((mask[:,:,np.newaxis],mask[:,:,np.newaxis]
                           #,mask[:,:,np.newaxis]
                           ), axis=2) 
    
    nuclei = pu.get_nuclei(img, data_loc, im_num, image_dims)      
    
    prediction = model.predict(x=[np.expand_dims(img, axis=0),np.expand_dims(mask, axis=0)])
    prediction = np.argmax(prediction[0,:,:], axis=2)  # this only shows values of (0,1), need to apply mask to get background
    if known_test ==True:
        prediction = np.argmax(np.load(y_dirs[im_num]), axis = 2)
        if record_metrics == True:
            eval_metrics = []
            eval_metrics.append(model.evaluate(x=[img[np.newaxis,],mask[np.newaxis,]],y=prediction[np.newaxis])) # might need to covert prediction to catigorical
    if save_img ==True:
        np.save(f'../evaluation/{model_name}/{im_num:04}_predict',np.array(prediction))
        
        
    avg_activation, num_cells = pu.det_activation(prediction, nuclei) # applies nuclei mask to predicction and determines activation for each cell
    data_set[im_num,0] = num_cells
    data_set[im_num,1] = avg_activation-1       # need to subtract 1 becasue acivation values range from 1-2










