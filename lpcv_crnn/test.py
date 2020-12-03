from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from warpctc_pytorch import CTCLoss
import os
from utils import misc
from data import dataset

import logging
import warnings
warnings.filterwarnings("ignore")

# Dependencies for pruning
from nni.compression.torch import L1FilterPruner, ADMMPruner

# Dependencies for finetuning
from models.pruned_crnn import CRNN_pruned
from models.crnn import CRNN

from utils.train_utils import validate_one_epoch
from train import get_logger

if __name__=='__main__':
    ''' 
    # Parse the arguments
    # Some useful parameters:
    # batchSize: Set the size of batch
    # expr_dir: Set saved directory path
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--valRoot', required=True, help='path to dataset for validation')
    # Set pretrain or finetune
    parser.add_argument('--ft', action='store_true', help='set finetune as true means use sample dataset to finetune the model')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batchSize', type=int, default=1024, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    # TODO(meijieru): epoch -> iter
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
    parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
    parser.add_argument('--expr_dir', default='./', help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadealta')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
    parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
    parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
    # Argument for finetuning
    parser.add_argument('--finetune', action='store_true', help='Whether to do finetuning')
    opt = parser.parse_args()
    # opt.alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # opt.alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-.:;<=>?@[]\\^_{}|~'
    print(opt)
    
    # Settings
    if not os.path.exists(opt.expr_dir):
        os.makedirs(opt.expr_dir)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    
    logger = get_logger('{}/exp.log'.format(opt.expr_dir))
    
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    transformer = dataset.resizeNormalize((100, 32))
    
    if opt.finetune:
        test_dataset = dataset.SampleDataset(root=opt.valRoot, transform=transformer)
    else:
        test_dataset = dataset.MJDataset(jsonpath=opt.valRoot, transform=transformer)
    
    logger.info('test dataset length:{}'.format(len(test_dataset)))
    assert test_dataset
        
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batchSize,
        shuffle=False, num_workers=int(opt.workers))

    nclass = len(opt.alphabet) + 1
    nc = 1

    converter = misc.strLabelConverter(opt.alphabet)
    criterion = CTCLoss()
    
    if opt.finetune:
        crnn = CRNN_pruned(opt.imgH, nc, nclass, opt.nh)
    else:
        crnn = CRNN(opt.imgH, nc, nclass, opt.nh)
    crnn.apply(misc.weights_init)

    if opt.pretrained != '':
        if logger:
            logger.info('loading pretrained model from %s' % opt.pretrained)
        else:
            print('loading pretrained model from %s' % opt.pretrained)
        
        tmp = torch.load(opt.pretrained)
        if 'module' in list(tmp.keys())[0]:
            crnn.load_state_dict(misc.load_multi(opt.pretrained), strict=True)
        else:
            crnn.load_state_dict(torch.load(opt.pretrained), strict=True)
        tmp = torch.load(opt.pretrained)
    logger.info(crnn)

    if opt.cuda:
        crnn.cuda()
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
        criterion = criterion.cuda()

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(crnn.parameters())
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

    logger.info('start testing!')
    
    validate_one_epoch(crnn, val_loader, criterion, converter, opt, logger)
