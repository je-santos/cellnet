import os
import tensorflow as tf
import argparse

import cell_utils as cu
import post_utils as pu


parser = argparse.ArgumentParser(description='MultiScaleNet properties.')
parser.add_argument('-nn', '--net_name',    default='base_78_w',   type=str)
parser.add_argument('-IS', '--image_set',    default='test_eval',  type=str)
parser.add_argument('--seed',               default=4980,          type=int)
parser.add_argument('--test',               default=0,             type=int)
parser.add_argument('--exp',                default=1,             type=int)
args = parser.parse_args()

model_name = args.net_name
model_seed = args.seed
image_set = '210119_2-165'
img_loc = f'../Images/{image_set}/*'
npy_loc = '../numpys/X_test_mean'
image_dims = (640, 560)           # (w x h)
scale = 1.683                     # square microns per pixel

filters_first_layer = 24
test_set = bool(args.test)
exp_data = bool(args.exp)
save_img = False

try:
    os.mkdir(f'../evaluation/{model_name}')
    print(f"Directory {model_name}  Created ")
except FileExistsError:
    print(f"Directory {model_name} already exists")

# load the model
try:
    model = tf.keras.models.load_model(f'models/{model_name}/{model_name}_{model_seed}',
                        custom_objects={'iou_loss': cu.iou_loss,
                                        'iou_0': cu.build_iou_for(label=0),
                                        'iou_1': cu.build_iou_for(label=1)})
    print('-'*50, f'\nSuccesfully loaded {model_name}_{model_seed}\n', '-'*50)
except:
    try:
        filepath = f'models/{model_name}/{model_name}_{model_seed}'
        model = cu.load_blank_model(filters_first_layer)
        tf.keras.Model.load_weights(model, filepath)
        print('-'*50, f'\nloaded empty {model_name}_{model_seed}\n', '-'*50)
    except:
        print('-'*50, '\nNo checkpoint found\n', '-'*50)

if test_set is True:
    # evaluate the model using a defined test set
    pu.test_evaluation(model=model, npy_loc=npy_loc, image_dims=image_dims,
                       model_name=model_name, model_seed=model_seed,
                       scale=scale, save_img=save_img)
if exp_data is True:
    # apply model to experimental data
    pu.cellnet_predict(model=model, img_set=image_set, img_loc=img_loc,
                       model_name=model_name, scale=scale,
                       image_dims=image_dims, save_img=save_img)
