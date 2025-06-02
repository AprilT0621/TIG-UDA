import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import nibabel as nib
from skimage import measure

def visda_acc(predict, all_label):
    matrix = confusion_matrix(all_label, predict)
    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    return aacc, acc

def soft_to_hard_pred(pred, channel_axis=1):
    """
    convert soft prediction to either 1 or 0.
    :param pred: the prediction
    :param channel_axis: the channel axis. For 'channel_first', it should be 1.
    :return: the 'hard' prediction
    """
    max_value = np.max(pred, axis=channel_axis, keepdims=True)
    return np.where(pred == max_value, 1, 0)

def to_categorical(mask, num_classes, channel='channel_first'):
    """
    convert ground truth mask to categorical
    :param mask: the ground truth mask
    :param num_classes: the number of classes
    :param channel: 'channel_first' or 'channel_last'
    :return: the categorical mask
    """
    if channel != 'channel_first' and channel != 'channel_last':
        assert False, r"channel should be either 'channel_first' or 'channel_last'"
    assert num_classes > 1, "num_classes should be greater than 1"
    unique = np.unique(mask)  #去除重复的并按升序的方式排序
    assert len(unique) <= num_classes, "number of unique values should be smaller or equal to the num_classes"
    assert np.max(unique) < num_classes, "maximum value in the mask should be smaller than the num_classes"
    if mask.shape[1] == 1:
        mask = np.squeeze(mask, axis=1)  #从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    if mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)
    eye = np.eye(num_classes, dtype='uint8') #将数组转成one-hot形式
    output = eye[mask]
    if channel == 'channel_first':
        output = np.moveaxis(output, -1, 1)   #交换2，3维度
    return output

def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded


def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y

def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    num_channel = mask.shape[1]
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in range(1, num_channel + 1):

        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)

        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img

def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everything needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header
    :param img_path: String with the path of the 'nii' or 'nii.gz' image file name.
    :return:Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def dice_coef(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    # return (2. * intersection / float(np.sum(y_true) + np.sum(y_pred)))
    return (2. * intersection + 1.0) / (np.sum(y_true) + np.sum(y_pred) + 1.0)

def dice_coef_multilabel(y_true, y_pred, numLabels=4, channel='channel_first'):
    """
    calculate channel-wise dice similarity coefficient
    :param y_true: the ground truth
    :param y_pred: the prediction
    :param numLabels: the number of classes
    :param channel: 'channel_first' or 'channel_last'
    :return: the dice score
    """
    assert channel=='channel_first' or channel=='channel_last', r"channel has to be either 'channel_first' or 'channel_last'"
    dice = 0
    if channel == 'channel_first':
        y_true = np.moveaxis(y_true, 1, -1)
        y_pred = np.moveaxis(y_pred, 1, -1)
    for index in range(1, numLabels):
        temp = dice_coef(y_true[..., index], y_pred[..., index])
        dice += temp

    dice = dice / (numLabels - 1) #平均dice
    return dice