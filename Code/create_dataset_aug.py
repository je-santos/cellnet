import numpy as np
import glob
import json
import os
import cv2
import tensorflow
from tensorflow.keras.utils import to_categorical
import post_utils as pu
import cell_utils as cu
from imgaug import augmenters as iaa
from sklearn.utils.class_weight import compute_class_weight

tif_loc = 'train'
set_name = 'train_w'
im_size = (640, 560)    # remember that this is wierd and backwards
train = True            # only set to True if creating the training set


def im_resize(X, im_size, intp='area'):
    if intp == 'area':
        a = cv2.resize(np.float32(X), im_size, interpolation=cv2.INTER_AREA)
    elif intp == 'nn':
        a = cv2.resize(np.float32(X), im_size, interpolation=cv2.INTER_NEAREST)
    return np.array(a).astype(np.uint8)


def load_data(tif_loc, im_size, imgs4train, mean_dic):
    im_loc_X = f'../Images/X_{tif_loc}'
    im_loc_y = f'../Images/Y_{tif_loc}'
    X_R = []
    X_G = []
    X_B = []
    y = []
    for im_num in range(imgs4train):
        print(f'Reading {im_num}')

        names_R = glob.glob(f'{im_loc_X}/*_{im_num:04}_R*.tif')[0]
        names_G = glob.glob(f'{im_loc_X}/*_{im_num:04}_G*.tif')[0]
        names_B = glob.glob(f'{im_loc_X}/*_{im_num:04}_B*.tif')[0]
        names_y = glob.glob(f'{im_loc_y}/*_{im_num:04}*.tif')[0]

        X_red = cv2.imread(names_R)
        X_green = cv2.imread(names_G)
        X_blue = cv2.imread(names_B)
        y_img = cv2.imread(names_y)

        X_R.append(im_resize(X_red[:,:,0], im_size))
        X_G.append(im_resize(X_green[:,:,0], im_size))
        X_B.append(im_resize(X_blue[:,:,0], im_size))

        y.append(im_resize(y_img[:,:,0], im_size, intp='nn'))

    X_R = np.array(X_R, dtype='float64')
    X_G = np.array(X_G, dtype='float64')
    X_B = np.array(X_B, dtype='float64')

    y = np.array(y)
    y[y == 128] = 1
    y[y == 255] = 2
    y[y > 2] = 0   # sanity check

    if len(np.unique(y)) > 3:
        raise NameError('The output has more than 3 values')
    X = np.concatenate(
            (np.expand_dims(X_R, 3),
             np.expand_dims(X_G, 3),
             np.expand_dims(X_B, 3)),
            3)

    masks = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    nuclei = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    try:
        os.mkdir(f'../numpys/Binary_R_{tif_loc}')
    except:
        print('Folder already exists')
    for im_num in range(np.shape(X)[0]):
        print(f'mask_{im_num:04}')
        img = X[im_num,:,:,:]
        masks[im_num,:,:] = pu.get_mask(img)
        nuclei[im_num,:,:] = pu.get_nuclei(img, mean_dic)

    return X, y, masks, nuclei


def save_data(tif_loc, X, y, masks, nuclei, img_num, num, sample_weights):
    try:
        os.mkdir(f'../numpys/X_{set_name}')
        os.mkdir(f'../numpys/y_{set_name}')
        os.mkdir(f'../numpys/Binary_R_{set_name}')
        os.mkdir(f'../numpys/weights_{set_name}')
    except: pass
    np.save(f'../numpys/X_{set_name}/{img_num:04}', X[num,:,:,:])
    np.save(f'../numpys/y_{set_name}/{img_num:04}', y[num,:,:,:])
    np.save(f'../numpys/Binary_R_{set_name}/{img_num:04}', masks[num,:,:])
    np.save(f'../numpys/weights_{set_name}/{img_num:04}', sample_weights[num])


def augmentation(X, y, masks, itterations):
    ys = np.concatenate((y[:,:,np.newaxis],y[:,:,np.newaxis],
                        y[:,:,np.newaxis]), axis=2)
    masks = np.concatenate((masks[:,:,np.newaxis],masks[:,:,np.newaxis],
                        masks[:,:,np.newaxis]), axis=2)
    B = np.concatenate((ys,X,masks), axis=2).astype(np.uint8)
    A = np.zeros((itterations, np.shape(B)[0], np.shape(B)[1], np.shape(B)[2]))

    for num in range(itterations):
        aug1 = iaa.Affine(scale={"x": (1, 1.5), "y": (1, 1.5)})
        aug2 = iaa.Fliplr(1)
        aug3 = iaa.Flipud(1)
        aug4 = iaa.ShearX((-20, 20), mode='reflect')
        aug5 = iaa.ShearY((-20, 20), mode='reflect')
        aug6 = iaa.Rotate((-45, 45), mode='reflect')
        aug7 = iaa.WithChannels((3, 4, 5), iaa.Multiply((0.5, 1.5)))
        aug_list = [aug1, aug2, aug3, aug4, aug5, aug6, aug7]
        ID = np.random.randint(0, len(aug_list))
        A[num,:,:,:] = aug_list[ID](image=B)
    aug_y       = A[:,:,:,0]
    aug_X       = A[:,:,:,3:6]
    aug_masks   = A[:,:,:,8]    
        
    return aug_X, aug_y, aug_masks


def create_data(tif_loc, im_size, imgs4train, itterations, want_aug):
    mean_dic = json.load(open('../numpys/mean_dic_train.json'))
    X, y, masks, nuclei = load_data(tif_loc, im_size, imgs4train, mean_dic)
    img_num = 0
    mean_R = mean_dic['mean_R']
    mean_G = mean_dic['mean_G']
    mean_B = mean_dic['mean_B']
    class_weights = compute_class_weight('balanced', np.unique(y),
                                         y.flatten())
    if want_aug is True:
        for count in range(imgs4train+1):
            if count == 0:
                aug_X, aug_y, aug_masks = X, y, masks
                aug_X = aug_X.astype(np.float16)
                aug_y = to_categorical(aug_y).astype(np.uint8)
            else:
                aug_X, aug_y, aug_masks = augmentation(X[count-1,:,:,:],
                                                       y[count-1,:,:],
                                                       masks[count-1,:,:],
                                                       itterations)
                aug_X = aug_X.astype(np.float16)
                aug_y = to_categorical(aug_y).astype(np.uint8)
            sample_weights = cu.get_image_weights(aug_y,
                                                  aug_masks[:,:,:,np.newaxis],
                                                  class_weights)
            aug_X[:,:,:,0] -= mean_R
            aug_X[:,:,:,1] -= mean_G
            aug_X[:,:,:,2] -= mean_B
            for num in range(np.shape(aug_X)[0]):
                save_data(tif_loc, aug_X, aug_y, aug_masks, nuclei, img_num,
                          num, sample_weights)
                print(f'saving_img_num_{img_num}')
                img_num += 1
            del aug_X, aug_y, aug_masks, sample_weights
    else:
        X = X.astype(np.float16)
        y = to_categorical(y).astype(np.uint8)
        sample_weights = cu.get_image_weights(y, masks[:,:,:,np.newaxis],
                                              class_weights)
        X[:,:,:,0] -= mean_R
        X[:,:,:,1] -= mean_G
        X[:,:,:,2] -= mean_B
        for num in range(np.shape(X)[0]):
            save_data(tif_loc, X, y, masks, nuclei, img_num, num,
                      sample_weights)
            img_num += 1
    print(f'{img_num}')


create_data(tif_loc, im_size, 20, 1, want_aug=False)
