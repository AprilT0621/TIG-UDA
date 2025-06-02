#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
from torch.nn import Softmax
import torch.nn as nn
import torch.nn.functional as F
from medpy.metric.binary import hd, dc, asd

def entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-10
    entropy = -input_ * torch.log2(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 


def im(outputs_test, gent=True):
    epsilon = 1e-10
    softmax_out = nn.Softmax(dim=1)(outputs_test)
    entropy_loss = torch.mean(entropy(softmax_out))
    if gent:
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log2(msoftmax + epsilon))
        entropy_loss -= gentropy_loss
    im_loss = entropy_loss * 1.0
    return im_loss


def adv(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(features.device)
    return torch.nn.BCELoss()(ad_out, dc_target)


def adv_local(features, ad_net, is_source=False):
    ad_out = ad_net(features).squeeze(4)
    batch_size = ad_out.size(0)
    num_heads = ad_out.size(1)
    seq_len = ad_out.size(2)
    seq_len2 = ad_out.size(3)
    # loss1 = torch.nn.BCELoss()
    if is_source:
        label = torch.from_numpy(np.array([[[[1]*seq_len]*seq_len2]*num_heads] * batch_size)).float().to(features.device)
    else:
        label = torch.from_numpy(np.array([[[[0]*seq_len]*seq_len2]*num_heads] * batch_size)).float().to(features.device)

    return ad_out, F.binary_cross_entropy_with_logits(ad_out,label)

def jaccard_loss(true, logits, eps=1e-7, activation=True):
    """
    Computes the Jaccard loss, a.k.a the IoU loss.
    :param true: a tensor of shape [B, H, W] or [B, C, H, W] or [B, 1, H, W].
    :param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
    :param eps: added to the denominator for numerical stability.
    :param activation: if apply the activation function before calculating the loss.
    :return: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        probas = F.softmax(logits, dim=1) if activation else logits

    true_1_hot = true.type(probas.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    probas = probas.contiguous()
    true_1_hot = true_1_hot.contiguous()
    intersection = probas * true_1_hot
    intersection = torch.sum(intersection, dims)
    cardinality = probas + true_1_hot
    cardinality = torch.sum(cardinality, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return 1 - jacc_loss

def evaluate(img_gt, img_pred, apply_hd=False, apply_asd=False):
    """
    Function to compute the metrics between two segmentation maps given as input.
    :param img_gt: Array of the ground truth segmentation map.
    :param img_pred: Array of the predicted segmentation map.
    :param apply_hd: whether to compute Hausdorff Distance.
    :param apply_asd: Whether to compute Average Surface Distance.
    :return: A list of metrics in this order, [dice myo, hd myo, asd myo, dice lv, hd lv asd lv, dice rv, hd rv, asd rv]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    dice_mean = []
    class_name = ["myo", "lv", "rv"]
    # Loop on each classes of the input images
    for c, cls_name in zip([1, 2, 3], class_name) :
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
        dice_mean.append(dice)
    dice_mean = np.mean(np.array(dice_mean))
    return dice_mean