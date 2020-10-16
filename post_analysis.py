# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:26:32 2020

@author: avh492
"""

'''
Things to adjust:
    - change averages that test images are normalized by (in create numpys code)
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
from tensorflow.keras.utils import to_categorical
from nd2reader import ND2Reader as nd2
import generalized_otsu as gotsu
import cell_utils as cu
import post_utils as pu
import pandas as pd

model_name          = 'cellnet_1014_dl_1655'
image_dims          = (640,560)                          # (w x h)
data_loc            = '../numpys/X_test'
npy_loc             ='../numpys/X_test/*.npy'   
scale               = 1.683 # square microns per pixel, from image metadata, remmeber to rescale when resizing the image

save_img        = False          # save an image of each prediction
record_metrics  = True
save_results    = True
num_heads       = 1             # 1-cells , 2-cells and nuclei



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


def test_evaluation(npy_loc, image_dims, model_name, record_metrics=True, save_img=True, save_results=True ):
    '''
    '''
    
    all_dirs = glob.glob(npy_loc)
    data_set = np.zeros((len(all_dirs),4))
    metric_names = ['loss', 'IOU_loss', 'IOU_0', 'IOU_1','# cells', 'activation', 'area', 'correct']
    eval_metrics = []
    for im_num in range( len(all_dirs) ):
        #for im_num in range(1):
        print(f'reading_image_{im_num:04}')
            
        img = np.zeros((image_dims[1],image_dims[0],3))
        img[:,:,:] = np.load( data_loc + f'/{im_num:04}.npy' )
    
        mask = pu.get_mask(img)
        mask = np.concatenate((mask[:,:,np.newaxis],mask[:,:,np.newaxis] ), axis=2) 
    
        nuclei = pu.get_nuclei(img) 
        nuclei = np.concatenate((nuclei[:,:,np.newaxis],nuclei[:,:,np.newaxis] ), axis=2)
        y_test = np.load(f'../numpys/y_test/{im_num:04}.npy')
        y_true = np.argmax(y_test, axis = 2)
        y_test = y_test[:,:,1:] 
        prediction = model.predict(x=[np.expand_dims(img, axis=0),np.expand_dims(mask, axis=0)])
        prediction = np.argmax(prediction[0,:,:], axis=2)           # This only predicts (0,1)
        prediction = np.add(prediction, (mask[:,:,0]/np.max(mask[:,:,0]))) # add the mask to segment out backgroud
        
        eval_metrics.append(model.evaluate(x=[np.expand_dims(img, axis=0),np.expand_dims(mask, axis=0)],y=y_test[np.newaxis]))
        
        if save_img ==True:
            plt.subplot(1,3,1)
            plt.imshow(y_true[:,:], clim=(0,2))
            plt.title('True')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1,3,2)
            plt.imshow(prediction[:,:], clim=(0,2))
            plt.title('Predicted')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1,3,3)
            plt.imshow( np.equal(prediction[:,:],y_true[:,:]), cmap = 'gray' )
            plt.title('Errors')
            plt.xticks([])
            plt.yticks([])
            fig = plt.gcf()
            fig.set_size_inches(20,6)
            plt.savefig(f'../evaluation/{model_name}/{im_num:04}.png')
        
        avg_activation, num_cells = pu.det_activation(prediction, nuclei[:,:,0]) # applies nuclei mask to predicction and determines activation for each cell
        area = np.count_nonzero(prediction)*scale
        num_correct, num_cells_comp = pu.compare_cells(nuclei[:,:,0], prediction, y_true)
        data_set[im_num,0] = num_cells
        data_set[im_num,1] = avg_activation
        data_set[im_num,2] = area    
        data_set[im_num,3] = num_correct
        
    eval_metrics = np.concatenate((eval_metrics,data_set), axis=1) 
    eval_metrics = pd.DataFrame(eval_metrics, columns = metric_names)
    if save_results ==True:
        eval_metrics.to_csv(f'../evaluation/{model_name}/eval_metrics.csv', header=True)







