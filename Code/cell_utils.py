import os
import datetime

import numpy as np
from shutil import copyfile

import tensorflow as tf
from tensorflow.keras.callbacks import TerminateOnNaN, ReduceLROnPlateau, \
                                        EarlyStopping, CSVLogger, \
                                        ModelCheckpoint
from tensorflow.keras.backend import cast, equal, argmax, epsilon, clip
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import build_resUnet as runet


def load_blank_model(filters_first_layer):
    loss = categorical_crossentropy
    optimizer = Adam(lr=1e-5)
    metrics = [iou_loss, build_iou_for(label=0), build_iou_for(label=1)]
    model = runet.build_cellnet(filters=filters_first_layer,
                                input_shape=(None, None, 3),
                                masks_shape=(None, None, 2),
                                nuclei_shape=(None, None, 2),
                                SnE=False, incl_attn=False,
                                incl_pvp=False, incl_masks=True)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics[:])
    return model


def get_data(data_loc, total_samples4train, image_dims):
    print('Loading the data')

    X_train = np.zeros((total_samples4train, image_dims[0], image_dims[1], 3))
    y_train = np.zeros((total_samples4train, image_dims[0], image_dims[1], 3))
    masks   = np.zeros((total_samples4train, image_dims[0], image_dims[1]))

    for im_num in range(total_samples4train):

        X_train[im_num,:,:,:] = np.load(f'{data_loc}/X_train/{im_num:04}.npy')
        y_train[im_num,:,:,:] = np.load(f'{data_loc}/y_train/{im_num:04}.npy')
        masks[im_num,:,:] = np.load(f'{data_loc}/Binary_R_train/{im_num:04}.npy')

    IDs = np.arange(X_train.shape[0])
    np.random.shuffle(IDs)
    X_train = X_train[IDs, :, :, :]
    y_train = y_train[IDs, :, :, :]
    masks = masks[IDs, :, :]

    # We are going to drop the first channel of y because we don't need
    # background info anymore
    y_train = y_train[:, :, :, 1:]
    masks /= 255
    masks = np.concatenate((masks[:,:,:,np.newaxis], masks[:,:,:,np.newaxis]),
                           axis=3)
    y_train = np.multiply(masks, y_train)
    return X_train, y_train, masks


def get_callbacks(model_name='tmp', model_seed=666, patience_LR=50,
                  patience_training=200):

    nan_terminate = TerminateOnNaN()
    ReduceLR = ReduceLROnPlateau(monitor='val_loss',
                                 factor=0.1, patience=patience_LR,
                                 verbose=1, mode='auto', min_delta=0.0001,
                                 cooldown=0, min_lr=0)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0,
                               patience=patience_training,
                               verbose=2, mode='auto', baseline=None,
                               restore_best_weights=True)

    csv_logger = CSVLogger(f'models/{model_name}/log_{model_seed}.csv',
                           append=True)

    checkpoint = ModelCheckpoint(f'models/{model_name}/{model_name}_{model_seed}',
                                 monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min',
                                 save_weights_only=True)

    return [early_stop, checkpoint, csv_logger, nan_terminate, ReduceLR]


def get_image_weights(y_train, masks, class_weights):

    masks = masks[:,:,:,0]
    masks = (masks == 0)*1
    y_train = np.concatenate((masks[:, :, :, np.newaxis], y_train), axis=3)
    y_integers = np.argmax(y_train, axis=3)
    d_class_weights = dict(enumerate(class_weights))
    sample_weight = np.zeros((np.shape(y_integers)[0]))
    for sample in range(np.shape(sample_weight)[0]):
        frac_0 = np.sum(y_integers[sample,:,:]==0 )/np.size(y_integers[sample,:,:])
        frac_1 = np.sum(y_integers[sample,:,:]==1 )/np.size(y_integers[sample,:,:])
        frac_2 = np.sum(y_integers[sample,:,:]==2 )/np.size(y_integers[sample,:,:])

        sample_weight[sample] = frac_0*d_class_weights[0] + \
            frac_1*d_class_weights[1] + frac_2*d_class_weights[2]

    return sample_weight


def make_folders(model_name, create_new_folder=True):
    try:
        os.mkdir(f'models/{model_name}')
        print("Directory ", model_name, " Created ")

    except FileExistsError:
        print("Directory ", model_name, " already exists")
        if create_new_folder is True:
            model_name = model_name+datetime.datetime.today().strftime("_%d_%H_%M")
            print("Creating ", model_name,  " directory")
            os.mkdir(f'models/{model_name}')

    try:
        os.remove(f'models/{model_name}' + '/traintest.py')
    except: 
        pass
    copyfile('traintest.py', f'models/{model_name}' + '/traintest.py')
    return model_name


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the
            normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = tf.keras.backend.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probs of each sample sum to 1
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = clip(y_pred, epsilon(), 1 - epsilon())
        # calc
        loss = y_true * tf.keras.backend.log(y_pred) * weights
        loss = -tf.keras.backend.sum(loss, -1)
        return loss

    return loss


def iou_loss(true, pred):  # this can be used as a loss if you make it negative
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return (tf.keras.backend.sum(intersection, axis=-1) +
            epsilon()) / (tf.keras.backend.sum(union, axis=-1) + epsilon())


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
    y_true = cast(equal(argmax(y_true), label), tf.keras.backend.floatx())
    y_pred = cast(equal(argmax(y_pred), label), tf.keras.backend.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = tf.keras.backend.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return tf.keras.backend.switch(tf.keras.backend.equal(union, 0), 1.0,
                                   intersection / union)


def build_iou_for(label: int, name: str = None):
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


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, file_loc, list_IDs,
                 batch_size, dim, num_heads,
                 n_channels_in=3, n_channels_out=2, shuffle=True):
        'Initialization'
        self.num_heads = num_heads
        self.dim = dim
        self.batch_size = batch_size
        self.file_loc = file_loc
        self.list_IDs = list_IDs
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y, sample_weights = self.__data_generation(list_IDs_temp,
                                                      self.num_heads)

        return X, y, sample_weights

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, num_heads):
        'Generates data containing batch_size samples'
        # Initialization
        X_train = np.empty((self.batch_size, *self.dim, self.n_channels_in))
        y = np.empty((self.batch_size, *self.dim, self.n_channels_out))
        masks = np.empty((self.batch_size, *self.dim, 2))
        sample_weights = np.ones((self.batch_size, 1))

        if num_heads == 2:
            nucleis = np.empty((self.batch_size, *self.dim, 2))
            y_nuc = np.empty((self.batch_size, *self.dim, self.n_channels_out))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            X_train[i, ] = np.load((self.file_loc+f'/X_train/{ID:04}.npy'))
            mask = np.load((self.file_loc+f'/Binary_R_train/{ID:04}.npy'))/255
            masks[i, ] = np.concatenate((mask[:,:,np.newaxis],
                                        mask[:,:,np.newaxis]), axis=2)

            y_train = np.load((self.file_loc+f'/y_train/{ID:04}.npy'))
            y[i, ]   = y_train[:,:,1:]
            y[i, ] = np.multiply(masks[i,],y[i,])
            weight = np.load((self.file_loc+f'/weights_train/{ID:04}.npy'))
            sample_weights[i, ] = sample_weights[i, ]*weight

            if num_heads == 2:
                nuclei = np.load((self.file_loc+f'/Binary_B_train/{ID:04}.npy'))/255
                nucleis[i, ] = np.concatenate((nuclei[:,:,np.newaxis],
                                             nuclei[:,:,np.newaxis]), axis=2)
                y_nuc[i, ] = np.multiply(nucleis[i, ], y[i, ])*2
                y_nuc[i, ][y_nuc[i, ] == 4] = 1
        # print(f'{sample_weights.shape}')
        if num_heads == 2:
            X = [X_train, masks, nucleis]
            y = [y, y_nuc]
        else:
            X = [X_train, masks]
        return X, y, sample_weights
