# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:26:32 2020

@author: avh492
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import skimage
from scipy.ndimage.measurements import label as find_conn_comps
import json
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from nd2reader import ND2Reader as nd2
import generalized_otsu as gotsu
import cell_utils as cu
import post_utils as pu
import pandas as pd

model_name          = 'cellnet_1014_dl_1655'
image_dims          = (640,560)                 # (w x h) 
img_loc             = '../Images/200818_2-141/*'
file_name           = 'test_set'
scale               = 1.683 # square microns per pixel, from image metadata, remmeber to rescale when resizing the image

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


def cellnet_predict(img_loc, file_name, image_dims, scale, num_heads = 1, save_img = False, save_results = True):
    '''
    - input is a folder of .nd2 images 
    - image_dims refers to the dimensions of the model NOT dimensions of the input image
    - scale is used to covert pixels to um, from image meta-data. Be sure to account for image resizing
    - output is a .csv file containing a table of # cells, % activation, and cell area for each image
    '''
    all_dirs = glob.glob(img_loc)
    data_set = np.zeros((len(all_dirs),3))
    col_names = ['# cells', 'activation', 'area']

    for im_num in range( len(all_dirs) ):
        print(f'reading_image_{im_num:04}')
        img = (pu.format_numpy(all_dirs,im_num, img_loc, image_dims)/256).astype(np.uint8)
    
        mask = pu.get_mask(img)
        mask = np.concatenate((mask[:,:,np.newaxis],mask[:,:,np.newaxis] ), axis=2)       # to match output dimesnions 
    
        nuclei = pu.get_nuclei(img) 
        nuclei = np.concatenate((nuclei[:,:,np.newaxis],nuclei[:,:,np.newaxis] ), axis=2) # to match output dimesnions    
    
        prediction = model.predict(x=[np.expand_dims(img, axis=0),np.expand_dims(mask, axis=0)])
        if num_heads == 2:
            prediction = prediction[0]
            nuclei_prediction = model.predict(x=[np.expand_dims(img, axis=0),np.expand_dims(nuclei, axis=0)])
            nuclei_prediction = nuclei_prediction[0]
            nuclei_prediction = np.argmax(nuclei_prediction[0,:,:], axis=2)
            nuclei_prediction = np.add(nuclei_prediction, (nuclei[:,:,0]/np.max(nuclei[:,:,0])))
        prediction = np.argmax(prediction[0,:,:], axis=2)           # This only predicts the cell classes
        prediction = np.add(prediction, (mask[:,:,0]/np.max(mask[:,:,0]))) # add the mask to add background class

        if save_img ==True:
            plt.imshow(prediction[:,:], clim=(0,2))
            plt.title('Predicted')
            plt.xticks([])
            plt.yticks([])
            fig = plt.gcf()
            fig.set_size_inches(20,6)
            plt.savefig(f'../evaluation/{model_name}/{im_num:04}.png')
        
        avg_activation, num_cells = pu.det_activation(prediction, nuclei[:,:,0]) 
        area = np.count_nonzero(prediction)*scale

        data_set[im_num,0] = num_cells
        data_set[im_num,1] = avg_activation
        data_set[im_num,2] = area    
        form_data = pd.DataFrame(data_set,index=all_dirs, columns=col_names)
    
    if save_results ==True:
        form_data.to_csv(f'../evaluation/{model_name}/{file_name}.csv',index=True, header=True)


cellnet_predict(img_loc, file_name, image_dims, scale, num_heads = 1, save_img = False, save_results = True)





