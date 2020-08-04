# -*- coding: utf-8 -*-
'''
To do:
        Subtract out means
        x and y do not have images in the same order
'''
from matplotlib import pyplot as plt
import numpy as np
import glob
from PIL import Image
import json
import os
import gc
import datetime
from shutil import copyfile

from tensorflow import keras

import cell_utils as cu


model_name          = 'cellnet_chan_0_21_18_47'
data_loc            = '../numpys'
image_dims          = (560,640)

try:
    os.mkdir(f'../evaluation/{model_name}')
    print("Directory " , model_name ,  " Created ")
except FileExistsError:
    print("Directory " , model_name ,  " already exists")

'''
load the model
'''

try:
    model = keras.models.load_model( 'models/' + model_name + '/' + model_name + '.h5', 
                                      custom_objects={'iou_loss': cu.iou_loss,
                                                     'iou_1': cu.build_iou_for(label=1)} ) #loads the model
    print('-'*50)
    print(f'Sucefully loaded {model_name}')    
    print('-'*50)

except:
  print('*'*50)
  print(f'No checkpoint found')
  print('*'*50)
  
'''
load the images
'''
print('Loading the data')
all_dirs = glob.glob(data_loc + '/X_test/*.npy')
#all_dirs = [1,2,3,4,5]        
X_test = np.zeros((len(all_dirs),image_dims[0],image_dims[1],3))
y_test = np.zeros((len(all_dirs),image_dims[0],image_dims[1],3))
        
for im_num in range( len(all_dirs) ):
    X_test[im_num,:,:,:] = np.load( data_loc + f'/X_test/{im_num:04}.npy' )
    y_test[im_num,:,:,:] = np.load( data_loc + f'/Y_test/{im_num:04}.npy' )

X_test = np.expand_dims(X_test[:,:,:,0], axis=3)
    
#y_test = to_categorical(y_test)
y_pred = model.predict(x=X_test[:,:,:,:]) 
y_true = np.argmax(y_test, axis=3)
y_pred = np.argmax(y_pred, axis=3)

'''
Evaluate model
'''
eval_metrics = []
for X, y in zip(X_test, y_test):
  eval_metrics.append(model.evaluate(x=X[np.newaxis,],y=y[np.newaxis,])) 
print(eval_metrics)
json.dump(f'{eval_metrics}', open(f'../evaluation/{model_name}/eval_metrics_{model_name}.json','w'))

'''
Plot images and predictions
'''
for test_num in range(len(all_dirs)):
  plt.subplot(1,3,1)
  plt.imshow(y_true[test_num,:,:], clim=(0,2))
  plt.title('True')
  plt.subplot(1,3,2)
  plt.imshow(y_pred[test_num,:,:], clim=(0,2))
  plt.title('Predicted')
  plt.subplot(1,3,3)
  plt.imshow( np.equal(y_pred[test_num,:,:],y_true[test_num,:,:]) )
  plt.title('Errors')
  #plt.figure(figsize=(180, 16), dpi= 80, facecolor='w', edgecolor='k')
  #plt.figure(figsize=(60, 60))
  #plt.rcParams['figure.figsize'] = [30, 30]


  im_acc = np.sum( np.equal(y_pred[test_num,:,:],y_true[test_num,:,:]) )/np.size(y_true[test_num,:,:])*100
  plt.suptitle(f'The accuracy is {im_acc}')

  fig = plt.gcf()
  fig.set_size_inches(20,6)
    
  plt.savefig(f'../evaluation/{model_name}/results_{test_num:04}.png')
            