# If having trouble use pip install keras-gpu

from matplotlib import pyplot as plt
import numpy as np
import time




import glob
from PIL import Image
import os



import tensorflow as tf
from tensorflow import keras
import sklearn

import keras.backend as K
K.clear_session()

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import TerminateOnNaN
from keras.callbacks import ReduceLROnPlateau

from keras.utils import to_categorical

from keras.optimizers import Adam

#!pip install livelossplot
from livelossplot.keras import PlotLossesCallback #liveloss plot has to be installed

import cell_utils as cu


"""
Semantic segmentation:
    x: tensor of size (batch, x, y, 3).
       the 3 stands for the channels in the RGB image.
       The images are standarized to have a mean of 0
       and a std of 1 (globally per channel)
    y: binary tensor of size (batch, x, y, categories).
       the number of categories is the number of desired classes.
       This can be achieved using to_categorical()
       
    TO DO:
        -ReduceonPlateau is blowing up
        -fix the thing to overwrite this .py file (erase it first perhaps)
        -sample weights don't seem to be working when retraining'
"""




model_name          = 'cellnet_test1'
create_new_folder   =  True

num_gpus            = 1
batch_size          = 8
total_samples4train = 876
epochs              = 500
image_dims          = (1080,1280)
validation_split    = 0.2    #splits the last x %
patience_training   = 55    #epochs before it stops training
filters_first_layer = 8
learning_rate       = 1e-4
use_dataloader      = False
data_loc            = '../train_scaled_np'



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
        all_dirs = glob.glob(data_loc + '/X/*.npy')
        
        X_train = np.zeros((len(all_dirs),image_dims[0],image_dims[1],3))
        y_train = np.zeros((len(all_dirs),image_dims[0],image_dims[1],3))
        
        for im_num in range( len(all_dirs) ):
            
            X_train[im_num,:,:,:] = np.load( data_loc + f'/X/{im_num}.npy' )
            y_train[im_num,:,:,:] = np.load( data_loc + f'/y/{im_num}.npy' )
            
        IDs = np.arange( X_train.shape[0] ) 
        np.random.shuffle( IDs )
        X_train = X_train[ IDs ,:,:,: ]
        y_train = y_train[ IDs ,:,:,: ]
        
        
        
"""
Callbacks and model internals
"""

loss    = tf.keras.losses.categorical_crossentropy
metrics = [ 'accuracy', cu.iou_loss, cu.build_iou_for(label=1) ] 

optimizer     = Adam(lr=learning_rate)
#plot_losses   = PlotLossesCallback( fig_path=(f'{model_name}/metrics.png'))  
plot_losses   = PlotLossesCallback( )  
nan_terminate = TerminateOnNaN()
ReduceLR      = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25,
                                  verbose=1, mode='auto', min_delta=0.0001, 
                                  cooldown=0, min_lr=0)
early_stop    = EarlyStopping(monitor='val_loss', min_delta=0, 
                                              patience=patience_training, 
                                              verbose=2, mode='auto', baseline=None,
                                              restore_best_weights=False)


csv_logger = CSVLogger(f'models/{model_name}/training_log.csv', append=True)

checkpoint = ModelCheckpoint(f'models/{model_name}/{model_name}.h5',
                             monitor='val_loss', verbose=1, save_best_only=True, 
                             mode='min',save_weights_only=False)

callbacks_list = [early_stop,checkpoint,csv_logger,nan_terminate,
                  #ReduceLR, 
                  plot_losses]

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
  
  from res_unet import *
  model = build_res_unet( input_shape  = ( None, None, 3 ), 
                         filters=filters_first_layer )
  
    
if num_gpus > 1:
    model = keras.utils.multi_gpu_model(model,gpus=num_gpus)
  


model.compile( loss=loss, optimizer=optimizer, metrics=metrics[:] )

start_time = time.time()

if use_dataloader == True:
    hist_model = model.fit_generator( generator=training_generator,
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
    
    hist_model = model.fit( x=X_train, y=y_train, epochs=epochs, batch_size=batch_size,
                       #validation_data = (x_val, y_val, val_sample_weights)
                       validation_split=validation_split, verbose=1, 
                       callbacks=callbacks_list, 
                       sample_weight= sample_weight,
                       shuffle=True )


elapsed_time = time.time() - start_time
print('Training time [hrs]: ', elapsed_time/3600)
#np.savetxt(("savedModels/%s/training_time.txt" % model_name),(np.expand_dims(elapsed_time/3600,0),np.expand_dims(elapsed_time,0)),delimiter=",", header="t [hrs], t[s]")
