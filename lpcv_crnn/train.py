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
import utils.misc as misc
from data import dataset

import logging
import warnings

warnings.filterwarnings("ignore")

# Dependencies for pruning
from nni.compression.torch import L1FilterPruner, ADMMPruner

# Dependencies for finetuning
from models.pruned_crnn import CRNN_pruned
from models.crnn import CRNN

from utils.train_utils import train_one_epoch

'''
Function for logger
'''


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


if __name__ == '__main__':
    ''' 
    # Parse the arguments
    # Some useful parameters:
    # batchSize: Set the size of batch
    # expr_dir: Set saved directory path
    # valInterval: Set how many intervals after we do evaluation
    # saveInterval: Set how many intervals after we save the model
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainRoot', required=True, help='path to dataset for training')
    parser.add_argument('--valRoot', required=True, help='path to dataset for validation')
    # Set pretrain or finetune
    parser.add_argument('--ft', action='store_true',
                        help='set finetune as true means use sample dataset to finetune the model')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batchSize', type=int, default=1024, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
    # TODO(meijieru): epoch -> iter
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
    parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
    parser.add_argument('--expr_dir', default='/data/yunhe/nni_crnn_finetune_pruned_l1_multi3_15',
                        help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
    parser.add_argument('--valInterval', type=int, default=400, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=400, help='Interval to be displayed')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadealta')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
    parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
    parser.add_argument('--random_sample', action='store_true',
                        help='whether to sample the dataset with random sampler')
    # Argument for pruning
    parser.add_argument('--prune', action='store_true', help='Whether to do pruning')
    # Argument for finetuning
    parser.add_argument('--finetune', action='store_true', help='Whether to do finetuning')
    opt = parser.parse_args()

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
        train_dataset = dataset.SampleDataset(root=opt.trainRoot)
        test_dataset = dataset.SampleDataset(root=opt.valRoot, transform=transformer)
    else:
        train_dataset = dataset.MJDataset(jsonpath=opt.trainRoot)
        test_dataset = dataset.MJDataset(jsonpath=opt.valRoot, transform=transformer)

    logger.info('train dataset length:{}'.format(len(train_dataset)))
    logger.info('test dataset length:{}'.format(len(test_dataset)))
    assert train_dataset

    if not opt.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                               shuffle=True, num_workers=int(opt.workers),
                                               collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW,
                                                                               keep_ratio=opt.keep_ratio))
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
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

        if not opt.finetune:
            crnn.load_state_dict(misc.load_multi(opt.pretrained), strict=True)
        else:
            crnn.load_state_dict(torch.load(opt.pretrained), strict=True)

    logger.info(crnn)

    if opt.cuda:
        crnn.cuda()
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
        # crnn = torch.nn.DataParallel(crnn, device_ids=[4, 5, 6, 7])
        criterion = criterion.cuda()

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(crnn.parameters())
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

    logger.info('start training!')

    # Pruning setting if opt.prune is set
    if opt.prune:
        configure_list = [{'sparsity': 0.5, 'op_types': ['Conv2d'], 'op_names': ['module.cnn.conv2']},
                          {'sparsity': 0.7, 'op_types': ['Conv2d'],
                           'op_names': ['module.cnn.conv3', 'module.cnn.conv4']},
                          {'sparsity': 0.85, 'op_types': ['Conv2d'],
                           'op_names': ['module.cnn.conv5', 'module.cnn.conv6']}]
        optimizer_finetune = optimizer
        pruner = L1FilterPruner(crnn, configure_list, optimizer_finetune)
        crnn = pruner.compress()

    # Train multi epoches
    for epoch in range(opt.nepoch):
        # Prune the model
        if opt.prune:
            pruner.update_epoch(epoch)
            if logger:
                logger.info('# Epoch {} #'.format(epoch))
            else:
                print('# Epoch {} #'.format(epoch))
            train_one_epoch(crnn, train_loader, val_loader, criterion, optimizer, converter, epoch, opt, logger,
                            callback=None)
            pruner.export_model(model_path='{}/pruned_fots{}.pth'.format(opt.expr_dir, epoch),
                                mask_path='{}/mask_fots{}.pth'.format(opt.expr_dir, epoch))
        # Train the model from scratch
        else:
            train_one_epoch(crnn, train_loader, val_loader, criterion, optimizer, converter, epoch, opt, logger,
                            callback=None)
