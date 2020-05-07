# If having trouble use pip install keras-gpu

from matplotlib import pyplot as plt
import numpy as np
import time
import datetime

import keras
from shutil import copyfile
import glob
from PIL import Image
import os
import tensorflow as tf
import sklearn

import keras.backend as K
K.clear_session()

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import TerminateOnNaN
from keras.callbacks import ReduceLROnPlateau

from keras.utils import to_categorical

#!pip install livelossplot
from livelossplot.keras import PlotLossesCallback #liveloss plot has to be installed

import cell_utils as cu


"""
Semantic segmentation:
    x: tensor of size (batch, x, y, 3).
       the 3 stands for the channels in the RGB image.
       The images are standarized to have a mean of 0
       and a std of 1
    y: binary tensor of size (batch, x, y, categories).
       the number of categories is the number of desired classes.
       This can be achieved using to_categorical()
"""




model_name = 'cellnet'
create_new_folder = False

num_gpus            = 1
batch_size          = 4
total_samples4train = 876
epochs              = 500
image_dims          = (1080,1280)
validation_split    = 0.2    #splits the last x %
patience_training   = 55    #epochs before it stops training
filters_first_layer = 8
learning_rate       = 1e-4
use_dataloader      = False
data_loc            = '../train_scaled_np'


try:
    os.mkdir(f'models/{model_name}')
    print("Directory " , model_name ,  " Created ") 
    
except FileExistsError:
    print("Directory " , model_name ,  " already exists")
    if create_new_folder == True:
      model_name = model_name + datetime.datetime.today().strftime("_%d_%m_%y_%H:%M:%S")
      print("Creating " , model_name ,  " directory")
      os.mkdir(f'models/{model_name}')

try: 
    copyfile('traintest.py',f'models/{model_name}' + '/traintest.py');
except:
    print('The script has not been modified')




if use_dataloader == True:
    # we create a number with the model name (unique)
    rnd_seed = np.sum( [ord(letter) for letter in model_name] )*123123
    # we use this num as the rnd seed for numpy
    np.random.seed( rnd_seed )
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
                                                    


all_dirs = glob.glob(data_loc + '/X/*.npy')

X_train = np.zeros((len(all_dirs),image_dims[0],image_dims[1],3)).astype(np.int8)
y_train = np.zeros((len(all_dirs),image_dims[0],image_dims[1],3)).astype(np.int8)

for im_num in range( len(all_dirs) ):
    
    X_train[im_num,:,:,:] = np.load( data_loc + f'/X/{im_num}.npy' ).astype(np.int16)
    y_train[im_num,:,:,:] = np.load( data_loc + f'/y/{im_num}.npy' ).astype(np.int16)
    

#y_train = to_categorical(y_train).astype(np.int8)

"""
Callbacks and model internals
"""

loss = tf.keras.losses.categorical_crossentropy
metrics=[ 'accuracy', cu.iou_loss, cu.build_iou_for(label=1) ] 

optimizer     = keras.optimizers.Adam(lr=learning_rate)
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

callbacks_list = [early_stop,checkpoint,plot_losses,csv_logger,nan_terminate,
                  ReduceLR]

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
  model = build_res_unet( input_shape  = ( None, None, 3 ), filters=filters_first_layer )
  
if num_gpus > 1:
    model = keras.utils.multi_gpu_model(model,gpus=num_gpus)
  
  model.compile( loss=loss, optimizer=optimizer, metrics=metrics[:] )





start_time = time.time()

if use_dataloader == True:
    hist_model = model.fit_generator( generator=training_generator,
                                      #steps_per_epoch=np.ceil(train_IDs.size / batch_size),
                                      #steps_per_epoch=1,
                                      epochs = epochs,
                                      verbose=1,
                                      validation_data=validation_generator,
                                      #validation_steps=np.ceil(val_IDs.size  / batch_size),
                                      callbacks=callbacks_list,
                                      use_multiprocessing=False,
                                      max_queue_size=4,
                                      #workers=2
                                      )
    
else:
    
    # This adds weight (attention) for balancing classes
    from sklearn.utils.class_weight import compute_class_weight
    
    y_integers    = np.argmax(y_train, axis=3)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers.flatten() )
    print(class_weights)
    d_class_weights = dict(enumerate(class_weights))
    
    sample_weight = np.zeros((np.shape(y_integers)[0])) 
    for sample in range(np.shape( sample_weight)[0] ):
      frac_0 = np.sum( y_integers[sample,:,:]==0 )/np.size(y_integers[sample,:,:])
      frac_1 = np.sum( y_integers[sample,:,:]==1 )/np.size(y_integers[sample,:,:])
      frac_2 = np.sum( y_integers[sample,:,:]==2 )/np.size(y_integers[sample,:,:])
    
      sample_weight[sample] = frac_0*d_class_weights[0]+frac_1*d_class_weights[1]+ \
                              frac_2*d_class_weights[2]
    
    del y_integers
    
    hist_model = model.fit( x=X_train, y=y_train, epochs=epochs, batch_size=batch_size,
                       #validation_data = (x_val, y_val, val_sample_weights)
                       validation_split=validation_split, verbose=1, 
                       callbacks=callbacks_list, 
                       sample_weight= sample_weight,
                       shuffle=True )


elapsed_time = time.time() - start_time
print('Training time [hrs]: ', elapsed_time/3600)
#np.savetxt(("savedModels/%s/training_time.txt" % model_name),(np.expand_dims(elapsed_time/3600,0),np.expand_dims(elapsed_time,0)),delimiter=",", header="t [hrs], t[s]")
