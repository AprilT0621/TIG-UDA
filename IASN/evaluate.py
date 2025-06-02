import numpy as np
import cv2
import torch
import pandas as pd
from tqdm import tqdm
import os
import glob
import imageio
import mcubes
from utils.utils import soft_to_hard_pred
from data.data_generate import to_categorical
from medpy.metric.binary import hd95, dc, asd, hd
import nibabel as nib
from PIL import Image
from albumentations import (
    Compose,
    CLAHE,
)
import SimpleITK as sitk

def crop_volume(vol, crop_size=112):
    """
    crop the images
    :param vol: the image
    :param crop_size: half size of cropped images
    :return:
    """
    return np.array(vol[:,
                    int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                    int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size])

def crop_volume2(vol, crop_size=112):
    """
    crop the images
    :param vol: the image
    :param crop_size: half size of cropped images
    :return:
    """
    return np.array(vol[
                    int(vol.shape[0] / 2) - crop_size: int(vol.shape[0] / 2) + crop_size,
                    int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size])


def reconstuct_volume(vol, crop_size=112, origin_size=256):
    """
    reconstruct the image (reverse process of cropping)
    :param vol: the images
    :param crop_size: half size of cropped images
    :param origin_size: the original size of the images
    :return:
    """
    recon_vol = np.zeros((len(vol), origin_size, origin_size, 4), dtype=np.float32)

    recon_vol[:,
    int(recon_vol.shape[1] / 2) - crop_size: int(recon_vol.shape[1] / 2) + crop_size,
    int(recon_vol.shape[2] / 2) - crop_size: int(recon_vol.shape[2] / 2) + crop_size, :] = vol

    return recon_vol

def read_img(pat_id, img_len, clahe=False):
    """
    read in raw images
    :param pat_id:
    :param img_len:
    :param clahe: whether to apply clahe (False)
    :return:
    """
    images = []
    for im in range(img_len):
        img = cv2.imread(os.path.join("/media/aprilt/Disk2/ljp/TIG-UDA/MSCMRSeg/trainB/pat_{}_lge_{}.png".format(pat_id, im)))
        if clahe:
            aug = Compose([CLAHE(always_apply=True)])
            augmented = aug(image=img)
            img= augmented["image"]
        images.append(img)
    return np.array(images)

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
    x = sitk.ReadImage(img_path)
    #print(x.GetSpacing())
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header, x.GetSpacing()

def resize_volume(img_volume, w=256, h=256):
    """
    :param img_volume:
    :return:
    """
    img_res = []
    for im in img_volume:
        img_res.append(cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_AREA))

    return np.array(img_res)

def get_csv_path(model_name, clahe=False):
    """
    generate csv path to save the result
    :param model_name:
    :param clahe:
    :return:
    """
    csv_path = model_name
    if clahe:
        csv_path += '_clahe'
    csv_path += '_evaluation.csv'
    return csv_path

def compute_metrics_on_files(gt, pred, spacing, ifhd=True, ifasd=True):
    """
    Function to give the metrics for two files
    :param gt: The ground truth image.
    :param pred: The predicted image.
    :param ifhd: whether to calculate HD.
    :param ifasd: whether to calculate ASD
    :return:
    """

    def metrics(img_gt, img_pred, spacing, ifhd=True, ifasd=True):
        """
        Function to compute the metrics between two segmentation maps given as input.

        img_gt: Array of the ground truth segmentation map.

        img_pred: Array of the predicted segmentation map.
        Return: A list of metrics in this order, [Dice endo, HD endo, ASD endo, Dice RV, HD RV, ASD RV, Dice MYO, HD MYO, ASD MYO]
        """

        if img_gt.ndim != img_pred.ndim:
            raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                             "same dimension, {} against {}".format(img_gt.ndim,
                                                                    img_pred.ndim))
        res = []
        for c in [500, 600, 200]:
            # Copy the gt image to not alterate the input
            gt_c_i = np.copy(img_gt)
            gt_c_i[gt_c_i != c] = 0

            # Copy the pred image to not alterate the input
            pred_c_i = np.copy(img_pred)
            pred_c_i[pred_c_i != c] = 0

            # Clip the value to compute the volumes
            gt_c_i = np.clip(gt_c_i, 0, 1)
            pred_c_i = np.clip(pred_c_i, 0, 1)
            # Compute the Dice
            dice = dc(gt_c_i, pred_c_i)
            # dice = dice_coef(gt_c_i, pred_c_i)

            h_d, a_sd = -1, -1
            if ifhd or ifasd:
                if np.sum(gt_c_i) == 0 or np.sum(pred_c_i) == 0:
                    dice = -1
                    h_d = -1
                    a_sd = -1
                else:
                    h_d = hd(gt_c_i, pred_c_i) if ifhd else h_d
                    a_sd = asd(gt_c_i, pred_c_i) if ifasd else a_sd
            res += [dice, h_d, a_sd]

        return res
    res = metrics(gt, pred, spacing, ifhd=ifhd, ifasd=ifasd)
    res_str = ["{:.3f}".format(r) for r in res]
    formatting = "LV {:>8} , {:>8} , {:>8} , RV {:>8} , {:>8} , {:>8} , Myo {:>8} , {:>8} , {:>8}"
    print(formatting.format(*res_str))
    return res

def evaluate_segmentation(bs=8, clahe=False, save=False, toprint=True, toplot=True, tocsv=True, model_name='',
                          ifhd=True, ifasd=True, pat_id_range=(6, 46), crop_size=224):
    """
    to evaluate the trained model
    :param unet_model: Name of the model to load.
    :param bs: batch size.
    :param clahe: whether to apply clahe.
    :param save: whether to save the evaluation result.
    :param toprint: whether to print the result.
    :param toplot: whether to plot the prediction.
    :param model_name: the model name (only for files to save).
    :param ifhd: whether to calculate HD.
    :param ifasd: whether to calculate ASD.
    :param pat_id_range: the pat_ids should be in (6, 46).
    :param weight_dir: the directory to the weight.
    :param crop_size: the size of the cropped images.
    :param klc: whether to apply 'keep largest connected component'.
    :return:
    """
    assert (pat_id_range[0] <= pat_id_range[1]) and (pat_id_range[0] >= 6) and (pat_id_range[1] <= 46), "pat_id_range error."
    if save:
        csv_path = get_csv_path(model_name=model_name, clahe=clahe)
        print(csv_path)
        if pat_id_range[0] > 3:
            df = pd.read_csv(csv_path)
        else:
            data = {'DSC': [], 'HD': [], 'ASD': [], 'cat': [], 'model': [], 'pad_id': []}
            df = pd.DataFrame(data)
    path = '/media/aprilt/Disk2/ljp/TIG-UDA/ssm_out1/out'
    result_path = os.path.join(path,'results')
    if tocsv or toplot:
        if not os.path.exists(result_path):
            os.makedirs(result_path)

    model = torch.load(os.path.join(path,'out_model_462.pt'))
    model = model.cpu()
    model.eval()
    ad_net = torch.load(os.path.join(path,'out_model_adv_462.pt'))
    ad_net = ad_net.cpu()
    ad_net.eval()
    if tocsv:
        df = pd.DataFrame(
            columns=['patient', 'dice_lv', 'dice_rv', 'dice_myo', 'hd_lv', 'hd_rv', 'hd_myo'])  # 列名
        df.to_csv(os.path.join(result_path, 'eval.csv'), index=False)  # 路径可以根据需要更改

    if toprint:
        endo_dc,myo_dc,rv_dc = [],[],[]
        endo_hd,myo_hd,rv_hd = [],[],[]
        endo_asd,myo_asd,rv_asd, = [],[],[]
    for pat_id in tqdm(range(pat_id_range[0], pat_id_range[1])):
        mask_path = "/media/aprilt/Disk2/ljp/TIG-UDA/MSCMRSeg/lge_test_gt/patient{}_LGE_manual.nii.gz".format(pat_id)
        nimg, affine, header, spacing = load_nii(mask_path)

        vol_resize= read_img(pat_id, nimg.shape[2], clahe=clahe)
        vol_resize = crop_volume(vol_resize, crop_size=crop_size // 2)
        x_batch = np.array(vol_resize, np.float32) / 255.
        x_batch = np.moveaxis(x_batch, -1, 1)
        pred = []
        for i in range(0, len(x_batch), bs):
            index = np.arange(i, min(i + bs, len(x_batch)))
            imgs = x_batch[index]
            pred1, _, _, _ = model(torch.tensor(imgs), ad_net, False)
            pred1 = pred1.cpu().detach().numpy()
            pred.append(pred1)
        pred = np.concatenate(pred, axis=0)
        pred = np.moveaxis(pred, 1, -1)
        pred = reconstuct_volume(pred, crop_size=112)
        p_save = np.argmax(pred, axis=3)
        p_save = np.array(p_save).astype(np.uint16)
        pred_resize = []
        for i in range(0, 4):
            pred_resize.append(resize_volume(pred[:, :, :, i], w=nimg.shape[0], h=nimg.shape[1]))
        pred = np.stack(np.array(pred_resize), axis=3)
        pred = np.argmax(pred, axis=3)
        masks = nimg.T
        pred = np.array(pred).astype(np.uint16)
        pred = np.where(pred == 1, 200, pred)
        pred = np.where(pred == 2, 500, pred)
        pred = np.where(pred == 3, 600, pred)
        if toplot:
            for j, (x, prediction, mask) in enumerate(zip(x_batch, p_save, masks)):
                prediction = np.where(prediction == 1, 85, prediction)
                prediction = np.where(prediction == 2, 212, prediction)
                prediction = np.where(prediction == 3, 255, prediction)
                p = Image.fromarray(prediction).convert('L')

                p.save(os.path.join(result_path, 'pat_{}_lge_{}.png'.format(pat_id,j)))
            print('plt finish')
        res = compute_metrics_on_files(masks, pred, spacing, ifhd=ifhd, ifasd=ifasd)
        if tocsv:
            list1 = [pat_id, res[0], res[3], res[6], res[1], res[4], res[7]]
            data = pd.DataFrame([list1])
            data.to_csv(os.path.join(result_path, 'eval.csv'), mode='a', header=False, index=False)
        endo_dc.append(res[0])
        rv_dc.append(res[3])
        myo_dc.append(res[6])
        if res[1] != -1:
            endo_hd.append(res[1])
        if res[4] != -1:
            rv_hd.append(res[4])
        if res[7] != -1:
            myo_hd.append(res[7])
        if res[2] != -1:
            endo_asd.append(res[2])
        if res[5] != -1:
            rv_asd.append(res[5])
        if res[8] != -1:
            myo_asd.append(res[8])
        mean_endo_dc = np.around(np.mean(np.array(endo_dc)), 3)
        mean_rv_dc = np.around(np.mean(np.array(rv_dc)), 3)
        mean_myo_dc = np.around(np.mean(np.array(myo_dc)), 3)
        std_endo_dc = np.around(np.std(np.array(endo_dc)), 3)
        std_rv_dc = np.around(np.std(np.array(rv_dc)), 3)
        std_myo_dc = np.around(np.std(np.array(myo_dc)), 3)
        print("Ave lv DC: {}, {}, Ave rv DC: {}, {}, Ave myo DC: {}, {}".format(mean_endo_dc, std_endo_dc, mean_rv_dc,
                                                                                  std_rv_dc, mean_myo_dc, std_myo_dc))
        print("Ave Dice: {:.3f}, {:.3f}".format((mean_endo_dc + mean_rv_dc + mean_myo_dc) / 3.,
                                                (std_endo_dc + std_rv_dc + std_myo_dc) / 3.))
        if ifhd:
            mean_endo_hd = np.around(np.mean(np.array(endo_hd)), 3)
            mean_rv_hd = np.around(np.mean(np.array(rv_hd)), 3)
            mean_myo_hd = np.around(np.mean(np.array(myo_hd)), 3)
            std_endo_hd = np.around(np.std(np.array(endo_hd)), 3)
            std_rv_hd = np.around(np.std(np.array(rv_hd)), 3)
            std_myo_hd = np.around(np.std(np.array(myo_hd)), 3)
            print("Ave lv HD: {}, {}, Ave rv HD: {}, {}, Ave myo HD: {}, {}".format(mean_endo_hd, std_endo_hd, mean_rv_hd, std_rv_hd, mean_myo_hd, std_myo_hd))
            print("Ave HD: {:.3f}, {:.3f}".format((mean_endo_hd + mean_rv_hd + mean_myo_hd) / 3., (std_endo_hd + std_rv_hd + std_myo_hd) / 3.))
        if ifasd:
            mean_endo_asd = np.around(np.mean(np.array(endo_asd)), 3)
            mean_rv_asd = np.around(np.mean(np.array(rv_asd)), 3)
            mean_myo_asd = np.around(np.mean(np.array(myo_asd)), 3)
            std_endo_asd = np.around(np.std(np.array(endo_asd)), 3)
            std_rv_asd = np.around(np.std(np.array(rv_asd)), 3)
            std_myo_asd = np.around(np.std(np.array(myo_asd)), 3)
            print("Ave lv ASD: {}, {}, Ave rv ASD: {}, {}, Ave myo ASD: {}, {}".format(mean_endo_asd, std_endo_asd, mean_rv_asd, std_rv_asd, mean_myo_asd, std_myo_asd))
            print("Ave ASD: {:.3f}, {:.3f}".format((mean_endo_asd + mean_rv_asd + mean_myo_asd) / 3., (std_endo_asd + std_rv_asd + std_myo_asd) / 3.))
        print(
            '{}, {}, {}, {}, {}, {}'.format(mean_myo_dc, std_myo_dc, mean_endo_dc, std_endo_dc, mean_rv_dc, std_rv_dc))
        if ifhd:
            print('{}, {}, {}, {}, {}, {}'.format(mean_myo_hd, std_myo_hd, mean_endo_hd, std_endo_hd, mean_rv_hd,
                                                  std_rv_hd))
        if ifasd:
            print('{}, {}, {}, {}, {}, {}'.format(mean_myo_asd, std_myo_asd, mean_endo_asd, std_endo_asd, mean_rv_asd,
                                                  std_rv_asd))
    print('Evaluation finished')

if __name__ == '__main__':
    evaluate_segmentation(bs=4, save=False, toprint=True, toplot = True, tocsv = False, ifhd=True,
                          ifasd=False, pat_id_range=(6, 46), crop_size=224)