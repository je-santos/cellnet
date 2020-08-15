# If having trouble use:
# conda create --name tf python=3.6 keras-gpu

from matplotlib import pyplot as plt
import numpy as np
import time



import glob
from PIL import Image
import os



import tensorflow as tf
import sklearn

#import tf.keras.backend as K
#K.clear_session()

#from tf.keras.callbacks import ModelCheckpoint
#from tf.keras.callbacks import CSVLogger
#from tf.keras.callbacks import EarlyStopping
#from tf.keras.callbacks import TerminateOnNaN
#from tf.keras.callbacks import ReduceLROnPlateau

#from tf.keras.utils import to_categorical

#pip install git+https://github.com/je-santos/livelossplot
from livelossplot import PlotLossesKeras #liveloss plot has to be installed

import cell_utils as cu

from livelossplot.plot_losses  import MatplotlibPlot



"""
Semantic segmentation:
    x: tensor of size (batch, x, y, 3).
       the 3 stands for the channels in the RGB image.
       The images are standarized to have a mean of 0 (globally per channel)
    y: binary tensor of size (batch, x, y, categories).
       the number of categories is the number of desired classes.
       This can be achieved using to_categorical()
       
    TO DO:
        -sample weights don't seem to be working when retraining'
"""




model_name          = 'cellnet_today'
create_new_folder   =  True

num_gpus            = 1
batch_size          = 8
total_samples4train = 34
epochs              = 500
image_dims          = (560,640)
validation_split    = 0.2    #splits the last x %
patience_training   = 100    #epochs before it stops training
patience_LR         = 75     #epochs before reducing LR
filters_first_layer = 8
learning_rate       = 1e-4
use_dataloader      = False
data_loc            = '../train_scaled_np'
channels            = [0,1,2]



model_name = cu.make_folders(model_name, create_new_folder)
# we create a number with the model name (unique)
rnd_seed = np.sum( [ord(letter) for letter in model_name] )*123123
# we use this num as the rnd seed for numpy
np.random.seed( rnd_seed )


if use_dataloader == True:
    
    ## Data-generator
    IDs = np.arange(0, total_samples4train)
    np.random.shuffle( IDs ) #get a mask to shuffle data
    val_split = int( total_samples4train*validation_split )
    train_IDs = IDs[:-val_split]
    val_IDs   = IDs[-val_split:]
    
    training_generator   = cu.DataGenerator(data_loc, train_IDs, 
                                            batch_size=batch_size,
                                            dim = image_dims)
    
    validation_generator = cu.DataGenerator(data_loc, val_IDs, 
                                            batch_size=batch_size,
                                            dim = image_dims)

else:
        print('Loading the data')
        all_dirs = glob.glob(data_loc + '/X_train/*.npy')
        
        X_train = np.zeros((len(all_dirs),image_dims[0],image_dims[1],3))
        y_train = np.zeros((len(all_dirs),image_dims[0],image_dims[1],3))
        masks   = np.zeros((len(all_dirs),image_dims[0],image_dims[1]))
        
        for im_num in range( len(all_dirs) ):
            
            X_train[im_num,:,:,:] = np.load( data_loc + f'/X_train/{im_num:04}.npy' )
            y_train[im_num,:,:,:] = np.load( data_loc + f'/y_train/{im_num:04}.npy' )
            masks[im_num,:,:]   = np.load( data_loc + f'/Binary_R_train/{im_num:04}.npy' )
            
        IDs = np.arange( X_train.shape[0] ) 
        np.random.shuffle( IDs )
        X_train = X_train[ IDs ,:,:,: ]
        y_train = y_train[ IDs ,:,:,: ]
        masks   = masks[   IDs ,:,:]
        
        
###### We are going to drop the first channel of y because we don't ned background info anymore
y_train = y_train[:,:,:,1:] 
masks/=255
masks = np.concatenate((masks[:,:,:,np.newaxis],masks[:,:,:,np.newaxis],
                        #masks[:,:,:,np.newaxis],masks[:,:,:,np.newaxis]
                        ), axis=3)

"""
Callbacks and model internals
"""

loss    = tf.keras.losses.categorical_crossentropy
metrics = [ cu.iou_loss, cu.build_iou_for(label=0) , ] 

optimizer     = tf.keras.optimizers.Adam(lr=learning_rate) 
plot_losses   = PlotLossesKeras(outputs=[MatplotlibPlot(figpath=f'models/{model_name}/metrics.png')]) 
nan_terminate = tf.keras.callbacks.TerminateOnNaN()
ReduceLR      = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                     factor=0.1, patience=patience_LR,
                                                     verbose=1, mode='auto', min_delta=0.0001, 
                                                     cooldown=0, min_lr=0)
early_stop    = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                                 patience=patience_training, 
                                                 verbose=2, mode='auto', baseline=None,
                                                 restore_best_weights=False)


csv_logger = tf.keras.callbacks.CSVLogger(f'models/{model_name}/training_log.csv', append=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(f'models/{model_name}/{model_name}.h5',
                             monitor='val_loss', verbose=1, save_best_only=True, 
                             mode='min',save_weights_only=False)

callbacks_list = [early_stop,checkpoint,csv_logger,nan_terminate,
                  ReduceLR, 
                  plot_losses]

try:
    model = tf.keras.models.load_model( 'models/' + model_name + '/' + model_name + '.h5', 
                                      custom_objects={'iou_loss': cu.iou_loss,
                                                     'iou_1': cu.build_iou_for(label=0)} ) #loads the model
    print('-'*50)
    print(f'Sucefully loaded {model_name}')    
    print('-'*50)

except:
  print('*'*50)
  print(f'No checkpoint found')
  print('*'*50)
  
  from res_unet import *
  model = build_res_unet( input_shape  = ( None, None, 3 ), 
                          masks_shape  = ( None, None, 2 ), 
                         filters=filters_first_layer )
  
    
if num_gpus > 1:
    model = tf.keras.utils.multi_gpu_model(model,gpus=num_gpus)
  


model.compile( loss=loss, optimizer=optimizer, metrics=metrics[:] )

start_time = time.time()

if use_dataloader == True:
    hist_model = model.fit( x=training_generator,
                                      epochs = epochs,
                                      verbose=1,
                                      validation_data=validation_generator,
                                      callbacks=callbacks_list,
                                      use_multiprocessing=False,
                                      max_queue_size=1,
                                      #workers=2
                                      )
    
else:

    #sample_weight = cu.get_image_weights(y_train)
    sample_weight = None
    
    hist_model = model.fit( x=[X_train,masks], y=y_train, epochs=epochs, batch_size=batch_size,
                       #validation_data = (x_val, y_val, val_sample_weights)
                       validation_split=validation_split, verbose=1, 
                       callbacks=callbacks_list, 
                       sample_weight= sample_weight,
                       shuffle=True )


elapsed_time = time.time() - start_time
print('Training time [hrs]: ', elapsed_time/3600)
#np.savetxt(("savedModels/%s/training_time.txt" % model_name),(np.expand_dims(elapsed_time/3600,0),np.expand_dims(elapsed_time,0)),delimiter=",", header="t [hrs], t[s]")
