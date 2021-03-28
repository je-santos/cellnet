import cv2
import numpy as np
import skimage
import glob
import pandas as pd
from scipy.ndimage.measurements import label as find_conn_comps
from nd2reader import ND2Reader as nd2
from matplotlib import pyplot as plt
from skimage import filters
import json


def normalize(img, mean_dic):
    # provides the mean correction necessary for application
    img = np.array(img, dtype='float64')
    img[:,:,0] -= 9  
    img[:,:,1] -= 8 
    img[:,:,2] -= 0 
    return img


def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
    img_8bit = img
    for i in range(3):
        scale = 255/(img[:,:,i].max() - img[:,:,i].min())
        bytedata = (img[:,:,i] - img[:,:,i].min())*scale
        img_8bit[:,:,i] = bytedata
    return img_8bit.astype(np.uint8)


def get_nuclei(image, mean_dic, test):
    '''
    - returns segmented nuclei from blue channel
    - assumes input image is formatted channel 1/2/3 = R/G/B
    '''

    img          = image[:,:,2]
    if test is True:
        img += mean_dic['mean_B']
    n, bin = np.histogram(img, bins=255)
    thresh, vals = GHT(n)     # breaks if histogram peak is 0 (mode of img = 0)
    if thresh <= np.min(img):
        thresh = skimage.filters.threshold_yen(img)
    tri = img < thresh
    image = 255*np.array(tri).astype(np.uint8)
    image = (image == 0).astype(int)
    nuclei, b = find_conn_comps(image)
    for comp in range(b):
        comp_size = np.sum(nuclei == comp)
        if comp_size < 60:  # 60 pixels ~100um2
            nuclei[nuclei == comp] = 0
        if comp_size > 1000:
            nuclei[nuclei == comp] = 0
    nuclei[nuclei > 0] = 255

    return nuclei


def get_mask(image):
    '''
    - returns segmented cells from red channel
    -assumes input image is formatted channel 1/2/3 = R/G/B
    - inverts image to remove noise from both background and within cells
    '''
    inv = []
    img = image[:,:,0]
    n, bin = np.histogram(img, bins=255)
    thresh1 = skimage.filters.threshold_triangle(img)
    tri = img <= thresh1
    image = 255*np.array(tri).astype(np.uint8)
    image = (image == 0).astype(int)
    a, b = find_conn_comps(image)
    for comp in range(b):
        comp_size = np.sum(a == comp)
        if comp_size < 200:  # 200 pixels is a good min cutoff for cells
            a[a == comp] = 0
    a[a > 0] = 255
    inv = a == 0
    inv = inv*255
    c, d = find_conn_comps(inv)
    for comp in range(d):
        comp_size = np.sum(c == comp)
        if comp_size < 10:
            inv[c == comp] = 0

    filled = inv == 0
    mask = filled*255

    return mask


def det_activation(prediction, nuclei):
    '''
    - overlays the nuclei mask on the prediction to determine activation state
    - assumes that 1 nuclei per cell, and that all cells have a nucleus
    - returns number of cells and avg activation of image
    '''
    nuclei_mask, num_comps = find_conn_comps(nuclei)
    activation_list = []
    for comp in range(num_comps):
        activation = prediction[nuclei_mask == comp+1]
        activation = np.rint(np.sum(activation) / len(activation))
        activation_list = np.append(activation_list, activation)

    activation_list = activation_list[activation_list != 0]
    avg_activation = np.average(activation_list) - 1  
    num_cells = len(activation_list)
    return avg_activation, num_cells


def format_numpy(all_dirs, im_num, img_loc, image_dims):
    '''
    - coverts image set to numpy arrays
    - Nikon microscope images saved as .nd2
    - These images are saved as channel 1/2/3 = B/R/G
    '''
    img = np.array(nd2(all_dirs[im_num]), dtype='float64')
    img_npy = np.zeros((image_dims[1], image_dims[0], 3))
    img_npy[:,:,0] = cv2.resize(img[1,:,:], image_dims, 
                                interpolation=cv2.INTER_AREA)
    img_npy[:,:,1] = cv2.resize(img[2,:,:], image_dims, 
                                interpolation=cv2.INTER_AREA)
    img_npy[:,:,2] = cv2.resize(img[0,:,:], image_dims, 
                                interpolation=cv2.INTER_AREA)
    return img_npy


def compare_cells(nuclei, prediction, Y):
    '''
    - compares results of prediction to y_true for test set
    - returns the number of cells as well as the number correctly predicted
    '''

    num_correct = 0
    nuc_mask, num_comps = find_conn_comps(nuclei)
    for comp in range(num_comps):
        predicted = prediction[nuc_mask == comp+1]
        truth = Y[nuc_mask == comp+1]
        avg_predicted = np.rint(np.sum(predicted) / len(predicted))
        avg_truth = np.rint(np.sum(truth) / len(truth))
        if avg_predicted == avg_truth:
            num_correct += 1

    return num_correct, num_comps


def metrics(nuclei, prediction, Y):
    '''
    - compares model prediction to truth
    - records number of True/False Positives/Negatives
    '''
    [num_correct, TN, FP, FN, TP] = [0, 0, 0, 0, 0]
    nuc_mask, num_comps = find_conn_comps(nuclei)
    for comp in range(num_comps):
        predicted = prediction[nuc_mask == comp+1]
        truth = Y[nuc_mask == comp+1]
        avg_predicted = np.rint(np.sum(predicted) / len(predicted))
        avg_truth = np.rint(np.sum(truth) / len(truth))
        if avg_predicted == avg_truth and avg_truth != 0:
            num_correct += 1
        if avg_predicted == 1 and avg_truth == 1:
            TN += 1
        if avg_predicted == 2 and avg_truth == 1:
            FP += 1
        if avg_predicted == 1 and avg_truth == 2:
            FN += 1
        if avg_predicted == 2 and avg_truth == 2:
            TP += 1

    return num_correct, TN, FP, FN, TP


def test_evaluation(model, npy_loc, image_dims, model_name, model_seed, scale,
                    save_img=True):
    '''
    - Uses test_set to compare model prediction to Truth
    - returns % cells correctly ID'd & other things
    '''
    mean_dic = json.load(open('../numpys/mean_dic_train.json'))
    all_dirs = glob.glob(f'{npy_loc}/*.npy')
    data_set = np.zeros((len(all_dirs), 8))
    metric_names = ['loss', 'IOU_loss', 'IOU_0', 'IOU_1', '# cells',
                    'activation', 'area', 'correct', 'TN', 'FP', 'FN', 'TP']
    eval_metrics = []
    for im_num in range(len(all_dirs)):
        print(f'reading_image_{im_num:04}')
        img = np.zeros((image_dims[1], image_dims[0], 3))
        img[:,:,:] = np.load(f'{npy_loc}/{im_num:04}.npy')

        mask = get_mask(img)
        mask = np.concatenate((mask[:,:,np.newaxis],mask[:,:,np.newaxis]),
                              axis=2)
        nuclei = get_nuclei(img, mean_dic, test=True)
        nuclei = np.concatenate((nuclei[:,:,np.newaxis],
                                 nuclei[:,:,np.newaxis]), axis=2)
        y_test = np.load(f'../numpys/y_test/{im_num:04}.npy')
        y_true = np.argmax(y_test, axis=2)
        y_test = y_test[:,:,1:]
        prediction = model.predict(x=[np.expand_dims(img, axis=0),
                                      np.expand_dims(mask, axis=0)])
        prediction = np.argmax(prediction[0,:,:], axis=2)
        prediction = np.add(prediction, (mask[:,:,0]/np.max(mask[:,:,0])))

        eval_metrics.append(model.evaluate(x=[np.expand_dims(img, axis=0),
                            np.expand_dims(mask, axis=0)],
                            y=y_test[np.newaxis], verbose=0))
        if save_img is True:
            plt.subplot(1, 3, 1)
            plt.imshow(y_true[:, :], clim=(0, 2))
            plt.title('True')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 3, 2)
            plt.imshow(prediction[:, :], clim=(0, 2))
            plt.title('Predicted')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 3, 3)
            plt.imshow(np.equal(prediction[:, :], y_true[:, :]), cmap='gray')
            plt.title('Errors')
            plt.xticks([])
            plt.yticks([])
            fig = plt.gcf()
            fig.set_size_inches(20, 6)
            plt.savefig(f'../evaluation/{model_name}/{im_num:04}.png')

        avg_activation, num_cells = det_activation(prediction, nuclei[:,:,0])
        area = np.count_nonzero(prediction)*scale
        num_correct, TN, FP, FN, TP = metrics(nuclei[:,:,0], prediction, y_true)
        if num_correct > num_cells:
            print('this bad')
        data_set[im_num, 0] = num_cells
        data_set[im_num, 1] = avg_activation
        data_set[im_num, 2] = area
        data_set[im_num, 3] = num_correct
        data_set[im_num, 4] = TN
        data_set[im_num, 5] = FP
        data_set[im_num, 6] = FN
        data_set[im_num, 7] = TP

    eval_metrics = np.concatenate((eval_metrics, data_set), axis=1)
    eval_metrics = pd.DataFrame(eval_metrics, columns=metric_names)
    eval_metrics.to_csv(f'../evaluation/{model_name}/Test_metrics_{model_seed}.csv',
                        header=True)


def cellnet_predict(model, img_set, img_loc, model_name, scale,
                    image_dims=(640, 560), num_heads=1, save_img=False):
    '''
    - input is a folder of .nd2 images
    - image_dims is dimensions of the model NOT of the input image
    - scale is used to covert pixels to um, from image meta-data
    - output is a .csv file containing a table of # cells, % activation
        and cell area for each image
    '''
    mean_dic = json.load(open('../numpys/mean_dic_train.json'))
    all_dirs = glob.glob(img_loc)
    data_set = np.zeros((len(all_dirs), 3))
    col_names = ['# cells', 'activation', 'area']

    for im_num in range(len(all_dirs)):
        print(f'reading_image_{im_num:04}')
        img16 = (format_numpy(all_dirs,im_num, img_loc, image_dims)).astype(np.float64)
        img8 = map_uint16_to_uint8(img16)
        mask = get_mask(img8)
        nuclei = get_nuclei(img8, mean_dic, test=False)
        img_norm = normalize(img16, mean_dic)
        mask = np.concatenate((mask[:,:,np.newaxis],
                               mask[:,:,np.newaxis]), axis=2)
        nuclei = np.concatenate((nuclei[:,:,np.newaxis],
                                 nuclei[:,:,np.newaxis]), axis=2)
        prediction = model.predict(x=[np.expand_dims(img_norm, axis=0),
                                      np.expand_dims(mask, axis=0)])
        if num_heads == 2:
            prediction = prediction[0]
            nuclei_prediction = model.predict(x=[np.expand_dims(img16, axis=0),
                                              np.expand_dims(nuclei, axis=0)])
            nuclei_prediction = nuclei_prediction[0]
            nuclei_prediction = np.argmax(nuclei_prediction[0,:,:], axis=2)
            nuclei_prediction = np.add(nuclei_prediction,
                                       (nuclei[:,:,0]/np.max(nuclei[:,:,0])))
        prediction = np.argmax(prediction[0,:,:], axis=2)
        prediction = np.add(prediction, (mask[:,:,0]/np.max(mask[:,:,0])))

        if save_img is True:
            plt.imshow(prediction[:, :], clim=(0, 2))
            plt.title('Predicted')
            plt.xticks([])
            plt.yticks([])
            fig = plt.gcf()
            fig.set_size_inches(20, 6)
            plt.savefig(f'../evaluation/{model_name}/{im_num:04}.png')

        avg_activation, num_cells = det_activation(prediction, nuclei[:,:,0])
        area = np.count_nonzero(prediction)*scale

        data_set[im_num, 0] = num_cells
        data_set[im_num, 1] = avg_activation
        data_set[im_num, 2] = area
        form_data = pd.DataFrame(data_set, index=all_dirs, columns=col_names)
        form_data.to_csv(f'../evaluation/{model_name}/{img_set}.csv',
                         index=True, header=True)


'''
Below from :
    Generalized Otsu's method
    @article{BarronECCV2020,
    Author = {Jonathan T. Barron},
    Title = {A Generalization of Otsu's Method and Minimum Error Thresholding},
    Journal = {ECCV},
    Year = {2020}
'''

csum = lambda z: np.cumsum(z)[: -1]
dsum = lambda z: np.cumsum(z[:: -1])[-2:: -1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties .
clip = lambda z: np.maximum(1e-30, z)


def preliminaries(n, x):
    """ Some math that is shared across each algorithm ."""
    assert np.all(n >= 0)
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    assert np.all(x[1:] >= x[: -1])
    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0 / (w0 + w1)
    p1 = w1 / (w0 + w1)
    mu0 = csum(n * x) / w0
    mu1 = dsum(n * x) / w1
    d0 = csum(n * x ** 2) - w0 * mu0 ** 2
    d1 = dsum(n * x ** 2) - w1 * mu1 ** 2
    return x, w0, w1, p0, p1, mu0, mu1, d0, d1


def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5):
    """ Our generalization of the above algorithms ."""
    assert nu >= 0
    assert tau >= 0
    assert kappa >= 0
    assert omega >= 0 and omega <= 1
    x, w0, w1, p0, p1, _, _, d0, d1 = preliminaries(n, x)
    v0 = clip((p0 * nu * tau ** 2 + d0) / (p0 * nu + w0))
    v1 = clip((p1 * nu * tau ** 2 + d1) / (p1 * nu + w1))
    f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa * omega) * np.log(w0)
    f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
    return argmax(x, f0 + f1), f0 + f1
