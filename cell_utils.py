import numpy as np
import glob
from PIL import Image
import json
import os
import gc
import datetime
from shutil import copyfile
from sklearn.utils.class_weight import compute_class_weight

#import keras 
#import keras.backend as K
#from tf.keras.utils import to_categorical

import tensorflow as tf


def get_image_weights(y_train):
    
    print('Calculating class weights')
    # This adds weight (attention) for balancing classes

    y_integers    = np.argmax(y_train, axis=3)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), 
                                                     y_integers.flatten() )
    
    print(f'The classes weights are : {class_weights}')
    d_class_weights = dict(enumerate(class_weights))
    
    sample_weight = np.zeros((np.shape(y_integers)[0])) 
    for sample in range(np.shape( sample_weight)[0] ):
      frac_0 = np.sum( y_integers[sample,:,:]==0 )/np.size(y_integers[sample,:,:])
      frac_1 = np.sum( y_integers[sample,:,:]==1 )/np.size(y_integers[sample,:,:])
      frac_2 = np.sum( y_integers[sample,:,:]==2 )/np.size(y_integers[sample,:,:])
    
      sample_weight[sample] = frac_0*d_class_weights[0]+frac_1*d_class_weights[1]+ \
                              frac_2*d_class_weights[2]
    
    return sample_weight



def make_folders(model_name, create_new_folder=True):
    try:
        os.mkdir(f'models/{model_name}')
        print("Directory " , model_name ,  " Created ") 
    
    except FileExistsError:
        print("Directory " , model_name ,  " already exists")
        if create_new_folder == True:
          model_name = model_name + datetime.datetime.today().strftime("_%d_%H_%M")
          print("Creating " , model_name ,  " directory")
          os.mkdir(f'models/{model_name}')
    
    
    try:
        os.remove(f'models/{model_name}' + '/traintest.py');
    except: pass
    copyfile('traintest.py',f'models/{model_name}' + '/traintest.py');


    return model_name




def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = tf.keras.backend.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probs of each sample sum to 1
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        loss = y_true * tf.keras.backend.log(y_pred) * weights
        loss = -tf.keras.backend.sum(loss, -1)
        return loss
    
    return loss

def iou_loss(true,pred):  #this can be used as a loss if you make it negative
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return (tf.keras.backend.sum(intersection, axis=-1) + tf.keras.backend.epsilon()) / (tf.keras.backend.sum(union, axis=-1) + tf.keras.backend.epsilon())

def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = tf.keras.backend.cast(tf.keras.backend.equal(tf.keras.backend.argmax(y_true), label), tf.keras.backend.floatx())
    y_pred = tf.keras.backend.cast(tf.keras.backend.equal(tf.keras.backend.argmax(y_pred), label), tf.keras.backend.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = tf.keras.backend.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return tf.keras.backend.switch(tf.keras.backend.equal(union, 0), 1.0, intersection / union)


def build_iou_for(label: int, name: str=None):
    """
    Build an Intersection over Union (IoU) metric for a label.
    Args:
        label: the label to build the IoU metric for
        name: an optional name for debugging the built method
    Returns:
        a keras metric to evaluate IoU for the given label
        
    Note:
        label and name support list inputs for multiple labels
    """
    # handle recursive inputs (e.g. a list of labels and names)
    if isinstance(label, list):
        if isinstance(name, list):
            return [build_iou_for(l, n) for (l, n) in zip(label, name)]
        return [build_iou_for(l) for l in label]

    # build the method for returning the IoU of the given label
    def label_iou(y_true, y_pred):
        """
        Return the Intersection over Union (IoU) score for {0}.
        Args:
            y_true: the expected y values as a one-hot
            y_pred: the predicted y values as a one-hot or softmax output
        Returns:
            the scalar IoU value for the given label ({0})
        """.format(label)
        return iou(y_true, y_pred, label)

    # if no name is provided, us the label
    if name is None:
        name = label
    # change the name of the method for debugging
    label_iou.__name__ = 'iou_{}'.format(name)

    return label_iou
        

def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = tf.keras.backend.int_shape(y_pred)[-1]
    # initialize a variable to store total IoU in
    total_iou = tf.keras.backend.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        total_iou = total_iou + iou(y_true, y_pred, label)
    # divide total IoU by number of labels to get mean IoU
    return total_iou / num_labels


# explicitly define the outward facing API of this module
__all__ = [build_iou_for.__name__, mean_iou.__name__]


# i=120;plt.imshow(X[i,:,:,0].squeeze().astype('float32'));plt.figure();plt.imshow(np.argmax(y[i,:,:,:],2))


def create_numpyset(im_loc_X = '../Images/X_train',
                    im_loc_y = '../Images/Y_train',
                    im_dir_X = '../train_scaled_np',
                    im_dir_y = '../train_scaled_np'):


    X_R     = []
    X_G     = []
    X_B     = []
    y       = []
    
    all_files = glob.glob(im_loc_X+'/*_B*')
    
    
    # load all the original images
    for im_num in range( len(all_files) ):
        print(f'Reading {im_num}')
        
        names_R = glob.glob(f'{im_loc_X}/*_{im_num:04}_R*.tif')[0]
        names_G = glob.glob(f'{im_loc_X}/*_{im_num:04}_G*.tif')[0]
        names_B = glob.glob(f'{im_loc_X}/*_{im_num:04}_B*.tif')[0]
        names_y = glob.glob(f'{im_loc_y}/*_{im_num:04}*.tif')[0]
        
        X_R.append(  np.array(Image.open(names_R)) )
        X_G.append(  np.array(Image.open(names_G)) )
        X_B.append(  np.array(Image.open(names_B)) )
        
        y.append(    np.array(Image.open(names_y)) )
        
    
    X_R = np.array(X_R, dtype='float64')
    X_G = np.array(X_G, dtype='float64')
    X_B = np.array(X_B, dtype='float64')
    
    y   = np.array(y)
    y[y==128] = 1
    y[y==255] = 2
    y[y>2   ] = 0   #sanity check
    
    if len( np.unique(y) ) > 3:
        raise NameError('The output has more than 3 values')
    
    #y = to_categorical(y).astype(np.int8)
    
    # towards data science says todo this
    mean_R = np.mean(X_R)
    mean_G = np.mean(X_G)
    mean_B = np.mean(X_B)
    
    mean_dic = {}
    mean_dic['mean_R']   = mean_R
    mean_dic['mean_G']   = mean_G
    mean_dic['mean_B']   = mean_B
    mean_dic['file_loc'] = im_dir_X
    
    
    X_R -= mean_R
    X_G -= mean_G
    X_B -= mean_B
    
    
    X = np.concatenate(
            ( np.expand_dims(X_R,3),
              np.expand_dims(X_G,3),
              np.expand_dims(X_B,3) ),
              3)
    
    X = np.concatenate((
                        X,
                        np.flip(X,1),
                        np.flip(X,2),
                        np.flip(np.flip(X,1),2)
                                    ),axis=0)
    
    
    y = np.concatenate((
                                y,
                        np.flip(y,1),
                        np.flip(y,2),
                        np.flip(np.flip(y,1),2)
                                    ),axis=0)

    
    
    X       = X.astype(np.float32)
    y       = to_categorical(y).astype(np.uint8)
    
    
    try:
        os.mkdir(im_dir_X + '/X' )
        os.mkdir(im_dir_y + '/y' )
    except: 
            print('folder exsists')
    
    for im_num in range( np.shape(X)[0] ):
        np.save( im_dir_X + '/X/' + f'{im_num}', X[im_num,:,:,:]  )
        np.save( im_dir_X + '/y/' + f'{im_num}', y[im_num,:,:,:]  )
        
    
    
    json.dump(str(mean_dic), open(im_dir_X + '/mean_dic.json','w'))
    
    

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, file_loc, list_IDs,  
                 batch_size, dim, 
                 n_channels_in=3,n_channels_out=3,shuffle=True):
        

        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.file_loc = file_loc
        self.list_IDs = list_IDs
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        #print(f'Hi, my len is {int(np.floor(len(self.list_IDs) / self.batch_size))}')
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
       'Generate one batch of data'
       # Generate indexes of the batch
       indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
       
       # Find list of IDs
       list_IDs_temp = [self.list_IDs[k] for k in indexes]

       # Generate data
       X, y = self.__data_generation(list_IDs_temp)
       
       return X, y
       
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
      
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
        

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels_in ))
        y = np.empty((self.batch_size, *self.dim, self.n_channels_out))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
    
            X[i,] = np.load((self.file_loc + '/X/' + str(ID) + '.npy'))
            
            y[i,] = np.load((self.file_loc + '/y/' + str(ID) + '.npy'))

        return X, y
