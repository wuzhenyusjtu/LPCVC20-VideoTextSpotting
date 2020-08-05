import argparse
import math

import numpy as np
import numpy.random as nprnd
import os
import torch
import torch.utils.data
import tqdm

import data as datasets
from fots_detector.model import FOTSModel

def load_multi(model_path):
    state_dict = model_path['model_state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict

def restore_checkpoint(folder, contunue, model_name=None):
    model = FOTSModel().to(torch.device("cuda"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=32, verbose=True, threshold=0.05, threshold_mode='rel')

    if model_name:
        checkppoint_name = os.path.join(folder, model_name)
    if os.path.isfile(checkppoint_name) and contunue:
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

def save_checkpoint(epoch, model, optimizer, lr_scheduler, best_score, folder, save_as_best):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if True:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_score': best_score
        }, os.path.join(folder, 'epoch_{}_checkpoint.pt'.format(epoch)))

    if save_as_best:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_score': best_score
        }, os.path.join(folder, 'best_checkpoint.pt'))
        print('Updated best_model')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'best_score': best_score
    }, os.path.join(folder, 'last_checkpoint.pt'))


def fill_ohem_mask(raw_loss, ohem_mask, num_samples_total, max_hard_samples, max_rnd_samples):
    h, w = raw_loss.shape
    if num_samples_total != 0:
        top_val, top_idx = torch.topk(raw_loss.view(-1), num_samples_total)
        num_hard_samples = int(min(max_hard_samples, num_samples_total))

        num_rnd_samples = max_hard_samples + max_rnd_samples - num_hard_samples
        num_rnd_samples = min(num_rnd_samples, num_samples_total - num_hard_samples)

        for id in range(min(len(top_idx), num_hard_samples)):
            val = top_idx[id]
            y = val // w
            x = val - y * w
            ohem_mask[y, x] = 1

        if num_rnd_samples != 0:
            for id in nprnd.randint(num_hard_samples, num_hard_samples + num_rnd_samples, num_rnd_samples):
                val = top_idx[id]
                y = val // w
                x = val - y * w
                ohem_mask[y, x] = 1


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
            ohem_cls_mask[batch_id, 0, shrunk_mask == 1] = 1
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


def fit(start_epoch, num_epochs, model, loss_func, opt, lr_scheduler, best_score, max_batches_per_iter_cnt, checkpoint_dir, train_dl, valid_dl):
    batch_per_iter_cnt = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        train_loss_stats = 0.0
        test_loss_stats = 0.0
        loss_count_stats = 0
        pbar = tqdm.tqdm(train_dl, 'Epoch ' + str(epoch), ncols=80)
        for cropped, classification, regression, thetas, training_mask in pbar:
            if batch_per_iter_cnt == 0:
                optimizer.zero_grad()
            prediction = model(cropped.to('cuda'))

            loss = loss_func(prediction, (classification, regression, thetas, training_mask)) / max_batches_per_iter_cnt
            train_loss_stats += loss.item()
            loss.backward()
            batch_per_iter_cnt += 1
            if batch_per_iter_cnt == max_batches_per_iter_cnt:
                opt.step()
                batch_per_iter_cnt = 0
                loss_count_stats += 1
                mean_loss = train_loss_stats / loss_count_stats
                pbar.set_postfix({'Mean loss': f'{mean_loss:.5f}'}, refresh=False)
        lr_scheduler.step(mean_loss, epoch)
        
        if valid_dl is None:
            val_loss = train_loss_stats / loss_count_stats
        else:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_loss_count = 0
                pbar = tqdm.tqdm(valid_dl, 'Val Epoch ' + str(epoch), ncols=80)
                loss_count_stats = 0
                cnt_error = 0
                for cropped, classification, regression, thetas, training_mask in pbar:
                    prediction = model(cropped.to('cuda'))
                    if prediction[0].shape[-1] != classification.shape[-1] or prediction[0].shape[-2] != classification.shape[-2]:
                        cnt_error += 1
                        continue
                    loss = loss_func(prediction, (classification, regression, thetas, training_mask))
                    val_loss += loss.item()
                    test_loss_stats += loss.item()
                    val_loss_count += len(cropped)
                    
                    batch_per_iter_cnt += 1
                    if batch_per_iter_cnt == max_batches_per_iter_cnt:
                        batch_per_iter_cnt = 0
                        loss_count_stats += 1
                        mean_loss = test_loss_stats / loss_count_stats
                        pbar.set_postfix({'Mean loss': f'{mean_loss:.5f}'}, refresh=False)
            val_loss /= val_loss_count
            print("{} shape not match".format(cnt_error))

        if best_score > val_loss:
            best_score = val_loss
            save_as_best = True
        else:
            save_as_best = False
        save_checkpoint(epoch, model, opt, lr_scheduler, best_score, checkpoint_dir, save_as_best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-folder-syn', type=str, required=True, help='Path to folder with syntext train images and labels')
    parser.add_argument('--train-folder-sample', type=str, required=True, help='Path to folder with sample train images and labels')
    parser.add_argument('--batch-size', type=int, default=21, help='Number of batches to process before train step')
    parser.add_argument('--batches-before-train', type=int, default=2, help='Number of batches to process before train step')
    parser.add_argument('--num-workers', type=int, default=4, help='Path to folder with train images and labels')
    parser.add_argument('--continue-training', action='store_true', help='Continue training')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus to use')
    parser.add_argument('--save-dir', type=str, default='./saved_models', help='Path to saved model dir')
    parser.add_argument('--pretrain-model', type=str, default='epoch_55_checkpoint.pt', help='Pretrained model name in save dir')
    parser.add_argument('--epochs', type=int, default=10, help='Num of epochs')
    parser.add_argument('--val', action='store_true', help='Use validation')
    args = parser.parse_args()

    data_set = datasets.MergeText(args.train_folder_syn, args.train_folder_sample, datasets.transform, datasets.transform)
    dl = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                     sampler=None, batch_sampler=None, num_workers=args.num_workers)
    if args.val:
        data_set_val = datasets.MergeTextVal(args.train_folder_syn, args.train_folder_sample, datasets.transform_finetune, datasets.transform_finetune)
        dl_val = torch.utils.data.DataLoader(data_set_val, batch_size=1, shuffle=True,
                                     sampler=None, batch_sampler=None, num_workers=args.num_workers)
    else:
        dl_val = None
    checkpoint_dir = args.save_dir
    if not os.path.exists(checkpoint_dir):
        print("{} dir does not exist. Created one.".format(checkpoint_dir))
        os.mkdir(checkpoint_dir)
    epoch, model, optimizer, lr_scheduler, best_score = restore_checkpoint(checkpoint_dir, args.continue_training, args.pretrain_model)
    if args.ngpus > 1:
        print('Use parallel')
        model = torch.nn.DataParallel(model)
    fit(epoch, args.epochs, model, detection_loss, optimizer, lr_scheduler, best_score, args.batches_before_train, checkpoint_dir, dl, dl_val)
