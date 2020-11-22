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


from standard import datasets
from standard.model import FOTSModel
from prune.model_pruned import FOTSModel_pruned
from modules.parse_polys import parse_polys
from utils.train_utils import load_multi, restore_checkpoint, fill_ohem_mask, detection_loss

# Train and validate the model
def val(model, loss_func, opt, max_batches_per_iter_cnt, valid_dl):
    batch_per_iter_cnt = 0
    test_loss_stats = 0.0
    loss_count_stats = 0
  
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_loss_count = 0
        pbar = tqdm.tqdm(valid_dl, 'Val result: ', ncols=80)
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
        print('Average val_loss: ', val_loss)
        print("{} shape not match".format(cnt_error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-folder-sample', type=str, required=True, help='Path to folder with sample test images and labels')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of batches to process before train step')
    parser.add_argument('--batches-before-train', type=int, default=2, help='Number of batches to process before train step')
    parser.add_argument('--num-workers', type=int, default=4, help='Path to folder with train images and labels')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus to use')
    parser.add_argument('--pretrain-model', type=str, required=True, help='Pretrained model path')
    parser.add_argument('--prune', action='store_true', help='Finetune pruned model')
    args = parser.parse_args()
    
    # Get dataloaders
    data_set_val = datasets.SampleDataset(args.test_folder_sample, datasets.transform, train=False)
    dl_val = torch.utils.data.DataLoader(data_set_val, batch_size=1, shuffle=True,
                                         sampler=None, batch_sampler=None, num_workers=args.num_workers)  
    
    epoch, model, optimizer, lr_scheduler, best_score = restore_checkpoint(True, args.pretrain_model, args.prune)
    if args.ngpus > 1:
        print('Use parallel')
        model = torch.nn.DataParallel(model)
    val(model, detection_loss, optimizer, args.batches_before_train, dl_val)

#     val(model, loss_func, opt, max_batches_per_iter_cnt, valid_dl)
