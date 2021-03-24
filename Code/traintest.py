import time

import numpy as np
import argparse

import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import cell_utils as cu
import build_resUnet as runet

"""
Semantic segmentation:
    x: tensor of size (batch, x, y, 3).
       the 3 stands for the channels in the RGB image.
       The images are standarized to have a mean of 0 (globally per channel)
    y: binary tensor of size (batch, x, y, categories).
       the number of categories is the number of desired classes.
       This can be achieved using to_categorical()
"""
# Don't pre-allocate memory; allocate as-needed
tf.config.experimental.set_memory_growth = True

# Argument parser
parser = argparse.ArgumentParser(description='MultiScaleNet properties.')

parser.add_argument('-nn', '--net_name',     default='base',        type=str)
parser.add_argument('--num_images',          default=20,            type=int)
parser.add_argument('-ffl', '--num_filters', default=2,             type=int)
parser.add_argument('-bs', '--Batch_size',   default=2,             type=int)
parser.add_argument('-LR', '--learn_rate',   default=1e-5,          type=float)
parser.add_argument('--seed',                default=222,           type=int)
parser.add_argument('--num_heads',           default=1,             type=int)
parser.add_argument('--masks',               default=1,             type=int)
parser.add_argument('-SNE',                  default=0,             type=int)
parser.add_argument('-attn',                 default=0,             type=int)
parser.add_argument('-pvp',                  default=0,             type=int)
args = parser.parse_args()

use_masks   = bool(args.masks)
use_SNE     = bool(args.SNE)
use_attn    = bool(args.attn)
use_pvp     = bool(args.pvp)
print(args)


# Basic model parameters
model_seed          = args.seed
model_name          = args.net_name
batch_size          = args.Batch_size
total_samples4train = args.num_images
filters_first_layer = args.num_filters
learning_rate       = args.learn_rate

epochs              = 1000
validation_split    = 0.2           # splits the last x %
patience_training   = 200           # epochs before it stops training
patience_LR         = 50            # epochs before reducing LR
image_dims          = (560, 640)

use_dataloader      = True
data_loc            = '../numpys'
create_new_folder   =  False


input_shape = (None, None, 3)
masks_shape = (None, None, 2)
nuclei_shape = (None, None, 2)


model_name = cu.make_folders(model_name, create_new_folder)
np.random.seed(model_seed)

'''
Generate the training data
'''
if use_dataloader is True:

    # Data-generator
    IDs = np.arange(0, total_samples4train)
    np.random.shuffle(IDs)
    val_split = int(total_samples4train*validation_split)
    train_IDs = IDs[:-val_split]
    val_IDs   = IDs[-val_split:]

    training_generator   = cu.DataGenerator(data_loc, train_IDs,
                                            batch_size=batch_size,
                                            dim=image_dims,
                                            num_heads=1)

    validation_generator = cu.DataGenerator(data_loc, val_IDs,
                                            batch_size=batch_size,
                                            dim=image_dims,
                                            num_heads=1)

else:
    X_train, y_train, masks = cu.get_data(data_loc, total_samples4train,
                                          image_dims)

"""
Callbacks and model internals
"""

try:
    model = load_model(f'models/{model_name}/{model_name}_{model_seed}',
                        custom_objects={'iou_loss': cu.iou_loss,
                                        'iou_0': cu.build_iou_for(label=0),
                                        'iou_1': cu.build_iou_for(label=1)})
    print('-'*50, f'\nSuccesfully loaded {model_name}\n', '-'*50)
except:
    print('-'*50, '\nNo checkpoint found\n', '-'*50)

# For use with multi-gpu machines
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    loss        = categorical_crossentropy
    optimizer   = Adam(lr=learning_rate)
    metrics     = [cu.iou_loss, cu.build_iou_for(label=0),
                   cu.build_iou_for(label=1)]
    callbacks_list = cu.get_callbacks(model_name, model_seed, patience_LR,
                                      patience_training)
    model = runet.build_cellnet(input_shape, masks_shape, nuclei_shape,
                                filters_first_layer, SnE=use_SNE,
                                incl_attn=use_attn, incl_pvp=use_pvp,
                                incl_masks=use_masks)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics[:],
                  sample_weight_mode="temporal")

    start_time = time.time()

'''
Train the model
'''
if use_dataloader is True:
    hist_model = model.fit(x=training_generator,
                           epochs=epochs,
                           verbose=1,
                           validation_data=validation_generator,
                           callbacks=callbacks_list,
                           use_multiprocessing=False,
                           max_queue_size=1,
                           )
else:
    sample_weight = None
    hist_model = model.fit(x=[X_train, masks], y=y_train,
                           epochs=epochs, batch_size=batch_size,
                           validation_split=validation_split, verbose=1,
                           callbacks=callbacks_list,
                           sample_weight=sample_weight,
                           shuffle=True)
tf.keras.Model.save_weights(model, f'models/{model_name}/{model_name}_{model_seed}')
elapsed_time = time.time() - start_time
print('Training time [hrs]: ', elapsed_time/3600)
