import math

import cv2
import numpy as np
import numpy.random as nprnd
import os
import torch
import torch.utils.data
import tqdm

import sys 
sys.path.append("..") 

# import datasets
from models.fots import FOTS
from models.pruned_fots import FOTS_pruned

# Load model trained with multi gpus
def load_multi(model_path):
    state_dict = model_path['model_state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict

# Restore model, optmizer, lr scheduler, score etc. from .pt file
def restore_checkpoint(contunue, model_name=None, prune=False):
    if prune:
        model = FOTS_pruned().to(torch.device("cuda"))
    else:
        model = FOTS().to(torch.device("cuda"))
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=32, verbose=True, threshold=0.05, threshold_mode='rel')
    if model_name:
        checkppoint_name = model_name
    if not prune:
        if contunue and os.path.isfile(checkppoint_name):
            print("Load checkpoint from {}".format(checkppoint_name))
            checkpoint = torch.load(checkppoint_name, map_location='cuda')
            model.load_state_dict(load_multi(checkpoint))
            epoch = 0
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            best_score = checkpoint['best_score']
            return epoch, model, optimizer, lr_scheduler, best_score
        else:
            return 0, model, optimizer, lr_scheduler, +math.inf
    else:
        if os.path.isfile(checkppoint_name) and contunue:
            print("Load checkpoint from {}".format(checkppoint_name))
            checkpoint = torch.load(checkppoint_name, map_location='cuda')
#             model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model.load_state_dict(checkpoint, strict=True)
            epoch = 0
            best_score = +math.inf
            return epoch, model, optimizer, lr_scheduler, best_score
        else:
            print('{} file not found'.format(checkppoint_name))
            return 0, model, optimizer, lr_scheduler, +math.inf

# Save checkpoint
def save_checkpoint(epoch, model, optimizer, lr_scheduler, best_score, folder, save_as_best):
    if not os.path.exists(folder):
        os.makedirs(folder)
    # if epoch > 60 and epoch % 6 == 0:
    if True:
        torch.save({
            'epoch': epoch,
#             'model_state_dict': model.module.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_score': best_score  # not current score
        }, os.path.join(folder, 'epoch_{}_checkpoint.pt'.format(epoch)))

    if save_as_best:
        torch.save({
            'epoch': epoch,
#             'model_state_dict': model.module.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_score': best_score  # not current score
        }, os.path.join(folder, 'best_checkpoint.pt'))
        print('Updated best_model')
    torch.save({
        'epoch': epoch,
#         'model_state_dict': model.module.state_dict(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'best_score': best_score  # not current score
    }, os.path.join(folder, 'last_checkpoint.pt'))

        
# Fill up ohem mask
def fill_ohem_mask(raw_loss, ohem_mask, num_samples_total, max_hard_samples, max_rnd_samples):
    h, w = raw_loss.shape
    if num_samples_total != 0:
        top_val, top_idx = torch.topk(raw_loss.view(-1), num_samples_total)
        num_hard_samples = int(min(max_hard_samples, num_samples_total))

        num_rnd_samples = max_hard_samples + max_rnd_samples - num_hard_samples
        num_rnd_samples = min(num_rnd_samples, num_samples_total - num_hard_samples)
        weight = num_hard_samples + num_rnd_samples

        for id in range(min(len(top_idx), num_hard_samples)):
            val = top_idx[id]
            y = val // w
            x = val - y * w
            ohem_mask[y, x] = 1 #/ weight

        if num_rnd_samples != 0:
            for id in nprnd.randint(num_hard_samples, num_hard_samples + num_rnd_samples, num_rnd_samples):
                val = top_idx[id]
                y = val // w
                x = val - y * w
                ohem_mask[y, x] = 1 #/ weight

# Calculate detection loss
def detection_loss(pred, gt):
    y_pred_cls, y_pred_geo, theta_pred = pred
    y_true_cls, y_true_geo, theta_gt, training_mask = gt
    y_true_cls, theta_gt = y_true_cls.unsqueeze(1), theta_gt.unsqueeze(1)
    y_true_cls, y_true_geo, theta_gt = y_true_cls.to('cuda'), y_true_geo.to('cuda'), theta_gt.to('cuda')

    raw_cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(input=y_pred_cls, target=y_true_cls, weight=None, reduction='none')

    d1_gt, d2_gt, d3_gt, d4_gt = torch.split(y_true_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred = torch.split(y_pred_geo, 1, 1)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_intersect = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
    h_intersect = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
    area_intersect = w_intersect * h_intersect
    area_union = area_gt + area_pred - area_intersect
    raw_tensor_loss = -torch.log((area_intersect+1) / (area_union+1)) + 10 * (1 - torch.cos(theta_pred - theta_gt))

    ohem_cls_mask = np.zeros(raw_cls_loss.shape, dtype=np.float32)
    ohem_reg_mask = np.zeros(raw_cls_loss.shape, dtype=np.float32)
    for batch_id in range(y_true_cls.shape[0]):
        y_true = y_true_cls[batch_id].squeeze().data.cpu().numpy().astype(np.uint8)
        mask = training_mask[batch_id].squeeze().data.cpu().numpy().astype(np.uint8)
        shrunk_mask = y_true & mask
        neg_mask = y_true.copy()
        neg_mask[y_true == 1] = 0
        neg_mask[y_true == 0] = 1
        neg_mask[mask == 0] = 0

        shrunk_sum = int(shrunk_mask.sum())
        if shrunk_sum != 0:
            ohem_cls_mask[batch_id, 0, shrunk_mask == 1] = 1 #/ shrunk_sum
        raw_loss = raw_cls_loss[batch_id].squeeze().data.cpu().numpy()
        raw_loss[neg_mask == 0] = 0
        raw_loss = torch.from_numpy(raw_loss)
        num_neg = int(neg_mask.sum())
        fill_ohem_mask(raw_loss, ohem_cls_mask[batch_id, 0], num_neg, 512, 512)

        raw_loss = raw_tensor_loss[batch_id].squeeze().data.cpu().numpy()
        raw_loss[shrunk_mask == 0] = 0
        raw_loss = torch.from_numpy(raw_loss)
        num_pos = int(shrunk_mask.sum())
        fill_ohem_mask(raw_loss, ohem_reg_mask[batch_id, 0], num_pos, 128, 128)

    ohem_cls_mask_sum = int(ohem_cls_mask.sum())
    ohem_reg_mask_sum = int(ohem_reg_mask.sum())
    if 0 != ohem_cls_mask_sum:
        raw_cls_loss = raw_cls_loss * torch.from_numpy(ohem_cls_mask).to('cuda')
        raw_cls_loss = raw_cls_loss.sum() / ohem_cls_mask_sum
    else:
        raw_cls_loss = 0

    if 0 != ohem_reg_mask_sum:
        raw_tensor_loss = raw_tensor_loss * torch.from_numpy(ohem_reg_mask).to('cuda')
        reg_loss = raw_tensor_loss.sum() / ohem_reg_mask_sum
    else:
        reg_loss = 0
    return reg_loss + raw_cls_loss