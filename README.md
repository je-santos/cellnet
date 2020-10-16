# Cellnet
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
 Model is run using tensorflow 2.3  
 
 1. Create a folder "Images" consisting of all images to process
 2. 
