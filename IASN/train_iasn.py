# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import math
import random
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from models.modeling import CONFIGS, AdversarialNetwork, UncertaintyDiscriminator
from models.dilated_unet import Segmentation_model
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.utils import soft_to_hard_pred, dice_coef_multilabel
from models import lossZoo
from data.data_generate import ImageProcessor, DataGenerator_PointNet
from torch.nn import BCELoss
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
CUDA_LAUNCH_BLOCKING = 1
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, epoch, is_adv=False):
    model_to_save = model.module if hasattr(model, 'module') else model
    if not is_adv:
        model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint.bin" % args.name)
        torch.save(model, os.path.join(args.output_dir, args.dataset, "%s_model_{}.pt".format(epoch) % args.name))
    else:
        model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint_adv.bin" % args.name)
        torch.save(model, os.path.join(args.output_dir, args.dataset, "%s_model_adv_{}.pt".format(epoch) % args.name))
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", os.path.join(args.output_dir, args.dataset))


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    model = Segmentation_model(filters=32, n_block=4, in_channels=3, n_class=args.num_classes, attention=True)
    model.load_from(np.load(args.pretrained_dir))
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_generators(args, ids_train, ids_valid, ids_train_lge, ids_valid_lge, batch_size=16, n_samples=1000,
                   crop_size=224):
    trainA_generator = DataGenerator_PointNet(df=ids_train, channel="channel_first", apply_noise=True, phase="train",  # True
                                              apply_online_aug=True, aug2=True,  # True True
                                              batch_size=batch_size, source="source", crop_size=crop_size,
                                              n_samples=n_samples, data_dir=args.data_dir)
    validA_generator = DataGenerator_PointNet(df=ids_valid, channel="channel_first", apply_noise=False, phase="valid",
                                              apply_online_aug=False,
                                              batch_size=batch_size, source="source", crop_size=crop_size,
                                              n_samples=-1, data_dir=args.data_dir)
    trainB_generator = DataGenerator_PointNet(df=ids_train_lge, channel="channel_first", apply_noise=True, # True
                                              phase="train",
                                              apply_online_aug=True, aug2=True,  # True True
                                              batch_size=batch_size, source="target", crop_size=crop_size,
                                              n_samples=n_samples, data_dir=args.data_dir)
    validB_generator = DataGenerator_PointNet(df=ids_valid_lge, channel="channel_first", apply_noise=False,
                                              phase="valid",
                                              apply_online_aug=False,
                                              batch_size=batch_size, source="target", crop_size=crop_size,
                                              n_samples=-1, data_dir=args.data_dir)
    return iter(trainA_generator), iter(validA_generator), iter(trainB_generator), iter(validB_generator)


def valid(args, model, ad_net, writer, data_generator, global_step, is_source):
    # Validation!
    eval_losses = AverageMeter()
    model.eval()
    ad_net.eval()
    dice_mean = []
    for batch in data_generator:
        x, y = batch
        x, y = torch.tensor(x).to(args.device), torch.tensor(y).to(args.device)
        with torch.no_grad():
            out, _, _, _ = model(x, ad_net=ad_net, is_source=is_source)
            eval_loss_seg1 = BCELoss()(torch.sigmoid(out), torch.tensor(y.clone(), dtype=torch.float32))
            eval_loss_seg2 = lossZoo.jaccard_loss(logits=torch.sigmoid(out), true=torch.tensor(y.clone(), dtype=torch.float32),
                                                  activation=False)
            eval_loss = eval_loss_seg1 + eval_loss_seg2
            eval_losses.update(eval_loss.item())
            out = out.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            y_pred = soft_to_hard_pred(out, 1)
            dice = dice_coef_multilabel(y_true=y, y_pred=y_pred, channel='channel_first')
            dice_mean.append(dice)
    dice_mean = np.mean(np.array(dice_mean))
    if is_source == True:
        print("Validating... (source dice=%f)" % dice_mean)
        writer.add_scalar("test/source_dice", scalar_value=dice_mean, global_step=global_step)
    else:
        print("Validating... (target dice=%f)" % dice_mean)
        writer.add_scalar("test/target_dice", scalar_value=dice_mean, global_step=global_step)
    return dice_mean, eval_loss

def train(args, model, n_samples):
    df = pd.DataFrame(
        columns=['loss', 'loss_seg', 'loss_ad_global', 'dice_s', 'dice_t'])  # 列名
    df.to_csv('/media/aprilt/Disk2/ljp/TIG-UDA/{}/{}/train.csv'.format(args.output_dir, args.dataset), index=False)  # 路径可以根据需要更改

    if args.local_rank in [-1, 0]:
        os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.dataset, args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    ids_train = ImageProcessor.split_data(os.path.join(args.data_dir, "aug_trainA.csv"))
    ids_valid = ImageProcessor.split_data(os.path.join(args.data_dir, "testA.csv"))
    ids_train_lge = ImageProcessor.split_data(os.path.join(args.data_dir, 'aug_trainB.csv'))
    ids_valid_lge = ImageProcessor.split_data(os.path.join(args.data_dir, 'testB.csv'))
    trainA_iterator, validA_iterator, trainB_iterator, validB_iterator = get_generators(args,
                                                                                        ids_train,
                                                                                        ids_valid,
                                                                                        ids_train_lge,
                                                                                        ids_valid_lge,
                                                                                        batch_size=args.train_batch_size,
                                                                                        n_samples=n_samples,
                                                                                        crop_size=224)

    config = CONFIGS[args.model_type]
    ad_net = UncertaintyDiscriminator(in_channel=args.num_classes)
    ad_net.to(args.device)
    ad_net_local = AdversarialNetwork(config.hidden_size // 12, config.hidden_size // 12)
    ad_net_local.to(args.device)
    # optimizer, lr, model, loss init
    optimizer_ad = torch.optim.SGD(ad_net.parameters(),
                                   lr=args.ad_lr,
                                   momentum=0.9,
                                   weight_decay=args.weight_decay)

    optimizer_adl = torch.optim.SGD(ad_net_local.parameters(),
                                    lr=args.adl_lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupCosineSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_adl = WarmupCosineSchedule(optimizer_adl, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupLinearSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_adl = WarmupLinearSchedule(optimizer_adl, warmup_steps=args.warmup_steps, t_total=t_total)

    model.zero_grad()
    ad_net.zero_grad()
    ad_net_local.zero_grad()

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    best_dice = 0
    smooth = 1e-7
    for global_step in range(1, t_total):
        model.train()
        ad_net.train()
        ad_net_local.train()
        loss_mean = []
        loss_seg_mean = []
        loss_ad_global_mean = []
        loss_ad_mean = []
        for (data_source, data_label), (data_target, _) in zip(trainA_iterator, trainB_iterator):
            x_s, y_s = torch.tensor(data_source).to(args.device), torch.tensor(data_label).to(args.device)
            x_t = torch.tensor(data_target).to(args.device)
            for param in ad_net.parameters():
                param.requires_grad = False
            for param in ad_net_local.parameters():
                param.requires_grad = False
            for param in model.parameters():
                param.requires_grad = True

            # 1. train the segmentation model with source images in supervised manner
            out_s, loss_ad_s, ad_out_s, encoder_s = model(x_s, ad_net_local, is_source=True)
            out_s = torch.sigmoid(out_s)
            loss_seg1 = BCELoss()(out_s, torch.tensor(data_label, dtype=torch.float32).to(args.device))
            loss_seg2 = lossZoo.jaccard_loss(logits=out_s,
                                             true=torch.tensor(data_label, dtype=torch.float32).to(args.device),
                                             activation=False)
            c = out_s.size()[1]
            uncertainty_mapS = -1.0 * out_s * torch.log(out_s + smooth) / math.log(c)
            loss_entropy = torch.mean(torch.sum(uncertainty_mapS, dim=1))
            loss_seg = loss_seg1 + loss_seg2 + args.w1 * loss_entropy
            loss_seg_mean.append((loss_seg).item())
            loss_seg.backward(retain_graph=True)

            # 2. train the segmentation model to fool the discriminators
            out_t, loss_ad_t, ad_out_t, encoder_t = model(x_t, ad_net_local, is_source=False)
            out_t = torch.sigmoid(out_t)
            adv_out_2 = ad_net(out_t.detach()) #
            loss_adv_global_t = F.binary_cross_entropy_with_logits(adv_out_2,
                                                                  torch.FloatTensor(adv_out_2.data.size()).fill_(1).to(
                                                                      args.device))
            loss_adv_t = F.binary_cross_entropy_with_logits(ad_out_t.detach(),
                                                                  torch.FloatTensor(ad_out_t.data.size()).fill_(1).to(
                                                                      args.device))
            loss_adv = 0.01 * loss_adv_global_t + 0.01 * loss_adv_t  # 0.01
            loss_adv.requires_grad_(True)
            loss_adv.backward(retain_graph=True)

            # 3. train the discriminators with images from source domain
            for param in ad_net.parameters():
                param.requires_grad = True
            for param in ad_net_local.parameters():
                param.requires_grad = True
            for param in model.parameters():
                param.requires_grad = False
            loss_ad_mean.append((loss_ad_s + loss_ad_t).item())
            ad_out = ad_net(out_s.detach())
            loss_ad_global_s = F.binary_cross_entropy_with_logits(ad_out,
                                                                  torch.FloatTensor(ad_out.data.size()).fill_(1).to(
                                                                      args.device))
            loss_ad_1 = 0.01 * loss_ad_global_s + 0.01 * loss_ad_s
            loss_ad_1.backward(retain_graph=True)

            # 4. train the discriminators with images from target domain
            ad_out_2 = ad_net(out_t.detach())
            loss_ad_global_t = F.binary_cross_entropy_with_logits(ad_out_2,
                                                                  torch.FloatTensor(ad_out_2.data.size()).fill_(0).to(
                                                                      args.device))
            loss_ad_2 = 0.01 * loss_ad_global_t + 0.01 * loss_ad_t
            loss_ad_2.backward()

            loss_ad_global = (loss_ad_global_s + loss_ad_global_t) / 2
            loss_ad_global_mean.append(loss_ad_global.item())
            loss = loss_seg + args.gamma * (loss_ad_s + loss_ad_t) + args.w2 * loss_ad_global
            loss_mean.append(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(ad_net.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(ad_net_local.parameters(), args.max_grad_norm)

            for param in ad_net.parameters():
                param.requires_grad = True
            for param in ad_net_local.parameters():
                param.requires_grad = True
            for param in model.parameters():
                param.requires_grad = True

            optimizer.step()
            optimizer.zero_grad()
            optimizer_ad.step()
            optimizer_ad.zero_grad()
            optimizer_adl.step()
            optimizer_adl.zero_grad()

            scheduler.step()
            scheduler_ad.step()
            scheduler_adl.step()

        loss_mean = np.mean(np.array(loss_mean))
        loss_seg_mean = np.mean(np.array(loss_seg_mean))
        loss_ad_global_mean = np.mean(np.array(loss_ad_global_mean))

        logger.info(" epoch = %d loss_seg = %f loss_ad_global = %f loss = %f", global_step, loss_seg_mean, loss_ad_global_mean, loss)
        if args.local_rank in [-1, 0]:
            writer.add_scalar("train/loss", scalar_value=loss_mean, global_step=global_step)
            writer.add_scalar("train/loss_seg", scalar_value=loss_seg_mean, global_step=global_step)
            writer.add_scalar("train/loss_ad_global", scalar_value=loss_ad_global_mean, global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)

        if (global_step + 1) % args.eval_every == 0 and args.local_rank in [-1, 0]:
            dice_s, loss1 = valid(args, model, ad_net_local, writer, validA_iterator, global_step, is_source=True)
            dice_t, loss2 = valid(args, model, ad_net_local, writer, validB_iterator, global_step, is_source=False)
            if dice_t > 0.9:
                save_model(args, model, epoch=global_step)
                save_model(args, ad_net_local, epoch=global_step, is_adv=True)
            if best_dice < dice_t:
                save_model(args, model, epoch=global_step)
                save_model(args, ad_net_local, epoch=global_step, is_adv=True)
                best_dice = dice_t
        if args.offdecay:
            if (global_step+1) % args.decay_e == 0:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.8
        list1 = [loss_mean, loss_seg_mean, loss_ad_global_mean, dice_s, dice_t]
        data = pd.DataFrame([list1])
        data.to_csv('/media/aprilt/Disk2/ljp/TIG-UDA/{}/{}/train.csv'.format(args.output_dir, args.dataset), mode='a', header=False, index=False)

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=False, default='out',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", help="Which downstream task.", default="supervised")
    parser.add_argument("--source_list", help="Path of the training data.")
    parser.add_argument("--target_list", help="Path of the test data.")
    parser.add_argument("--test_list", help="Path of the test data.")
    parser.add_argument("--num_classes", default=4, type=int,
                        help="Number of classes in the dataset.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/imagenet21k_ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="mscmr_out", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--beta", default=0.1, type=float,  #0.1
                        help="The importance of the adversarial loss.")
    parser.add_argument("--gamma", default=0.01, type=float, #0.01
                        help="The importance of the local adversarial loss.")
    parser.add_argument("--theta", default=0.1, type=float, #0.1
                        help="The importance of the IM loss.")
    parser.add_argument("--use_im", default=False, action="store_true",
                        help="Use information maximization loss.")
    parser.add_argument("--msa_layer", default=12, type=int,
                        help="The layer that incorporates local alignment.")
    parser.add_argument("--is_test", default=False, action="store_true",
                        help="If in test mode.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,  # 3e-2
                        help="The initial learning rate for SGD.")
    parser.add_argument("--ad_lr", default=3e-4, type=float,  # 3e-6
                        help="The initial learning rate for SGD.")
    parser.add_argument("--adg_lr", default=3e-3, type=float,  # 3e-6
                        help="The initial learning rate for SGD.")
    parser.add_argument("--adl_lr", default=3e-3, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=1000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,  #42
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--data_dir', help="the directory to the data", type=str, default='/media/aprilt/Disk2/ljp/TIG-UDA/MSCMRSeg/')  # 加
    parser.add_argument('--w1', help="the directory to the data", type=float, default=1.0)
    parser.add_argument('--w2', help="the directory to the data", type=float, default=1.0)
    parser.add_argument('--offdecay', help="whether not to use learning rate decay for unet", default=True)
    parser.add_argument('--decay_e', help="the epochs to decay the unet learning rate", type=int, default=200)

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    set_seed(args)

    args, model = setup(args)
    model.to(args.device)
    train(args, model, n_samples=670)


if __name__ == "__main__":

    main()

