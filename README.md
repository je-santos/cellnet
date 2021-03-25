# A Deep learning approach to myofibroblast identification and segmentation
## Introduction
<img align="right" width="40%" height="40%" src="https://github.com/ahillsley/cellnet/blob/master/illustrations/Picture1.png"/>
Cellnet is a convolutional neural network trained to classify and segment 2 types of cells; activated myofibroblasts and fibroblasts. 

The model input is a standard 3 channel flourescent image:

1. actin
2. alpha-smooth muscle actin
3. Nuclei / DAPI

The output is a segmented image with 3 classes:

1. Activated Myofibroblast
2. Nonactivated fibroblast
3. Background


## Model Architecture
<img align="right" width="40%" height="40%" src="https://github.com/ahillsley/cellnet/blob/master/illustrations/Picture2.png"/>
The base architecture of cellnet is a Unet with the encoder and decoder each consisting of 3 "res blocks". With each resblock the number of filters in the Conv2D layer is doubled and the image downscaled by a factor of 2. There are also 3 skip connections that pass information directly from the encoder to the decoder, which help preserve spacial information. Lastly a binary mask of the cells is made from the actin channel and passed directly to the decoder in order to segment the cells from the background.

## Instructions
   
### Code

  - **application**: for use applying a trained model to any data_set, can read images saved as numpy arrays or .nd2
  - **cell_utils**: general utility functions for cellnet, including: model callbacks, metrics, and data generators
  - **create_dataset_aug**: Used to generate training datasets including image augmentation
  - **post_utils**: utility functions for application including cell mask and nuclei segmentation
  - **resunet**: functions used to build the model structure of cellnet
  - **traintest**: Used to train all models
