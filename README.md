# A Deep learning approach to myofibroblast identification and segmentation
## Introduction
<img align="right" width="40%" height="40%" src="https://github.com/ahillsley/cellnet/blob/master/illustrations/Picture1.png"/>
Cellnet is a convolutional neural network trained to classify and segment 2 types of cells; activated myofibroblasts and fibroblasts. 

The model input is a 3 channel flourescent image:
1. F-actin
2. alpha-smooth muscle actin
3. Nuclei / DAPI

The output is a segmented image with 3 classes:
1. Activated Myofibroblast
2. Nonactivated fibroblast
3. Background

Postprocessing provides the following for each image: 
1. Number of cells
2. average activation of all cells 
3. average area of all cells

## Model Architecture
<img align="right" width="40%" height="40%" src="https://github.com/ahillsley/cellnet/blob/master/illustrations/Picture2.png"/>
The base architecture of cellnet is a ResUnet with the encoder and decoder each consisting of 3 residual blocks. The model takes advantage of the F-actin channel of the input to segment a cell mask. This is then applied to the end of the decoder, effectivly eliminating the background class and reducing the number of predicted classes from 3 to 2. 

### Code

  - **application**: for use applying a trained model to any data_set, can read images saved as numpy arrays or .nd2
  - **cell_utils**: general utility functions for cellnet, including: model callbacks, metrics, and data generators
  - **create_dataset_aug**: Used to generate training datasets including image augmentation
  - **post_utils**: utility functions for application including cell mask and nuclei segmentation
  - **build_Resunet**: functions used to build the model structure of cellnet
  - **traintest**: Used to train all models, options to include attention, pvp, and squeeze and excite layers as well as a second output head
