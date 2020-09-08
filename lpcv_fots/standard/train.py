import argparse
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

import datasets
from model import FOTSModel
from prune.model_pruned import FOTSModel_pruned
from modules.parse_polys import parse_polys

from utils.train_utils import load_multi, restore_checkpoint, save_checkpoint, fill_ohem_mask, detection_loss

# Train and validate the model
def fit(start_epoch, num_epochs, model, loss_func, opt, lr_scheduler, best_score, max_batches_per_iter_cnt, checkpoint_dir, train_dl, valid_dl):
    batch_per_iter_cnt = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        train_loss_stats = 0.0
        test_loss_stats = 0.0
        loss_count_stats = 0
        pbar = tqdm.tqdm(train_dl, 'Epoch ' + str(epoch), ncols=80)
        cnt_error_t = 0
        for cropped, classification, regression, thetas, training_mask in pbar:
            if batch_per_iter_cnt == 0:
                optimizer.zero_grad()
            prediction = model(cropped.to('cuda'))
            
            if prediction[0].shape[-1] != classification.shape[-1] or prediction[0].shape[-2] != classification.shape[-2]:
                cnt_error_t += 1
                continue
            
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
    parser.add_argument('--train-folder-sample', type=str, default=None, help='Path to folder with sample train images and labels')
    parser.add_argument('--batch-size', type=int, default=21, help='Number of batches to process before train step')
    parser.add_argument('--batches-before-train', type=int, default=2, help='Number of batches to process before train step')
    parser.add_argument('--num-workers', type=int, default=4, help='Path to folder with train images and labels')
    parser.add_argument('--continue-training', action='store_true', help='Continue training')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus to use')
    parser.add_argument('--save-dir', type=str, default='./saved_models', help='Path to saved model dir')
    parser.add_argument('--pretrain-model', type=str, default=None, help='Pretrained model path')
    parser.add_argument('--epochs', type=int, default=10, help='Num of epochs')
    parser.add_argument('--val', action='store_true', help='Use validation')
    parser.add_argument('--prune', action='store_true', help='Finetune pruned model')
    args = parser.parse_args()
    
    # Get dataloaders
    dl_val = None
    if args.train_folder_sample:
        # Merge SynthText dataset with sample dataset we generated from LPCV videos, use merged dataset for training and testing
        print('Use Merged dataset')
        # Set train as True to use data augmentation while training
        data_set = datasets.MergeText(args.train_folder_syn, args.train_folder_sample, datasets.transform, train=True)
        dl = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                     sampler=None, batch_sampler=None, num_workers=args.num_workers)
        if args.val:
            # Set train as False to disbale data augmentation while validating
            data_set_val = datasets.MergeText(args.train_folder_syn, args.train_folder_sample, datasets.transform, train=False)
            dl_val = torch.utils.data.DataLoader(data_set_val, batch_size=1, shuffle=True,
                                         sampler=None, batch_sampler=None, num_workers=args.num_workers)            
    else:
        # Only use SynthText dataset
        print('Use SynthText dataset')
        # Set train as True to use data augmentation while training
        data_set = datasets.SynthText(args.train_folder_syn, datasets.transform, train=True)
        dl = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                     sampler=None, batch_sampler=None, num_workers=args.num_workers)
        if args.val:
            # Set train as False to disbale data augmentation while validating
            data_set_val = datasets.SynthText(args.train_folder_syn, datasets.transform, train=False)
            dl_val = torch.utils.data.DataLoader(data_set_val, batch_size=1, shuffle=True,
                                         sampler=None, batch_sampler=None, num_workers=args.num_workers)

    checkpoint_dir = args.save_dir
    if not os.path.exists(checkpoint_dir):
        print("{} dir does not exist. Created one.".format(checkpoint_dir))
        os.mkdir(checkpoint_dir)
    epoch, model, optimizer, lr_scheduler, best_score = restore_checkpoint(args.continue_training, args.pretrain_model, args.prune)
    if args.ngpus > 1:
        print('Use parallel')
        model = torch.nn.DataParallel(model)
    fit(epoch, args.epochs, model, detection_loss, optimizer, lr_scheduler, best_score, args.batches_before_train, checkpoint_dir, dl, dl_val)
