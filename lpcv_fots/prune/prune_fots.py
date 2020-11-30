import sys 
sys.path.append("..") 

from models.fots import FOTS
import argparse

import torch.utils.data

from data import datasets
from train import train_one_epoch
from nni.compression.torch import L1FilterPruner
from utils.train_utils import detection_loss, load_multi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-folder-syn', type=str, required=True, help='Path to folder with syntext train images and labels')
    parser.add_argument('--train-folder-sample', type=str, default=None, help='Path to folder with sample train images and labels')
    parser.add_argument('--batch-size', type=int, default=21, help='Number of batches to process before train step')
    parser.add_argument('--batches-before-train', type=int, default=2, help='Number of batches to process before train step')
    parser.add_argument('--num-workers', type=int, default=4, help='Path to folder with train images and labels')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus to use')
    parser.add_argument('--save-dir', type=str, default='./saved_models', help='Path to saved model dir')
    parser.add_argument('--pretrain-model', type=str, default=None, help='Pretrained model name in save dir')
    parser.add_argument('--epochs', type=int, default=10, help='Num of epochs')
    parser.add_argument('--val', action='store_true', help='Use validation')
    args = parser.parse_args()
    
    model = FOTS().to(torch.device("cuda"))
    model = model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=32, \
                                                              verbose=True, threshold=0.05, threshold_mode='rel')

    checkpoint_name = args.pretrain_model
    checkpoint = torch.load(checkpoint_name, map_location='cpu')
    model.load_state_dict(load_multi(checkpoint))

    epoch = 0
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    best_score = checkpoint['best_score']

    configure_list = [{
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
            'op_names': [
                'resnet.conv1','resnet.layer1.0.conv1','resnet.layer1.0.conv2','resnet.layer1.1.conv1','resnet.layer1.1.conv2',\
                'resnet.layer1.2.conv1', 'resnet.layer1.2.conv2', 'resnet.layer2.0.conv1', 'resnet.layer2.0.conv2', \
                'resnet.layer2.0.downsample.0', 'resnet.layer2.1.conv1', 'resnet.layer2.1.conv2', 'resnet.layer2.2.conv1', \
                'resnet.layer2.2.conv2', 'resnet.layer2.3.conv1', 'resnet.layer2.3.conv2', 'resnet.layer3.0.conv1', \
                'resnet.layer3.0.conv2', 'resnet.layer3.0.downsample.0', 'resnet.layer3.1.conv1', 'resnet.layer3.1.conv2', \
                'resnet.layer3.2.conv1', 'resnet.layer3.2.conv2', 'resnet.layer3.3.conv1', 'resnet.layer3.3.conv2', \
                'resnet.layer3.4.conv1', 'resnet.layer3.4.conv2', 'resnet.layer3.5.conv1', 'resnet.layer3.5.conv2', \
                'resnet.layer4.0.conv1', 'resnet.layer4.0.conv2', 'resnet.layer4.0.downsample.0', 'resnet.layer4.1.conv1', \
                'resnet.layer4.1.conv2', 'resnet.layer4.2.conv1', 'resnet.layer4.2.conv2',\
                'center.0.0', 'center.1.0',  'decoder4.squeeze.0', 'decoder3.squeeze.0', 'decoder2.squeeze.0','decoder1.squeeze.0', \
            ]
        }]
                     
    # Prune model and test accuracy without fine tuning.
    # print('=' * 10 + 'Test on the pruned model before fine tune' + '=' * 10)
    optimizer_finetune = optimizer
    pruner = L1FilterPruner(model, configure_list, optimizer_finetune)
    model = pruner.compress()

    # Code for fots training
    train_folder_syn = args.train_folder_syn
    train_folder_sample = args.train_folder_sample
    output_path = args.save_dir
    data_set = datasets.MergeText(train_folder_syn, train_folder_sample, datasets.transform, train=True)
    dl = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                         sampler=None, batch_sampler=None, num_workers=args.num_workers)
    dl_val = None
    if args.val:
        data_set_val = datasets.MergeText(train_folder_syn, train_folder_sample, datasets.transform, train=False)
        dl_val = torch.utils.data.DataLoader(data_set_val, batch_size=1, shuffle=True,
                                                 sampler=None, batch_sampler=None, num_workers=args.num_workers)        
    max_batches_per_iter_cnt = 2

    for epoch in range(50):
        pruner.update_epoch(epoch)
        print('# Epoch {} #'.format(epoch))
        val_loss = train_one_epoch(model, detection_loss, optimizer, lr_scheduler, max_batches_per_iter_cnt, dl, dl_val, epoch)
        pruner.export_model(model_path='{}/pruned_fots{}.pth'.format(output_path, epoch), mask_path='{}/mask_fots{}.pth'.format(output_path, epoch))
