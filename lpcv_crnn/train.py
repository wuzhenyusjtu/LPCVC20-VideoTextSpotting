from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset

import models.crnn as crnn
import copy

# from crnn_prune import CRNN as CRNN_p
import logging
import warnings
warnings.filterwarnings("ignore")

# Dependencies for pruning
from nni.compression.torch import L1FilterPruner, ADMMPruner
import time

# Dependencies for finetuning
from models.pruned_crnn import CRNN_pruned as CRNN_p
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

'''
Initialize model

Args:
    m: model to be initialized
'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
'''
Load model from model trained with DP (multi GPUs)

Args:
    model_path: path to the weight of the model

Returns:
    A new state dicts copied from the weight of the model.
'''
def load_multi(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict

# Function for validation
def val(net, dataset, criterion, logger=None, max_iter=100):
    if logger:
        logger.info('Start val')
    else:
        print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        if logger:
            logger.info('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
        else:
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    if logger:
        logger.info('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    else:
        print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

# Function for training one batch
def trainBatch(net, criterion, optimizer, train_iter):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

# Train one epoch
def train(crnn, train_loader, criterion, optimizer, epoch, opt, logger=None, callback=None):
    train_iter = iter(train_loader)
    i = 0
#     print("Start CRNN training.")
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer, train_iter)
        loss_avg.add(cost)
        i += 1
        if i % opt.displayInterval == 0:
            if logger:
                logger.info('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            else:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            val(crnn, test_dataset, criterion, logger=logger)

        # do checkpointing
        if i % opt.saveInterval == 0:
            torch.save(
                crnn.state_dict(), '{}/netCRNN_{}_{}.pth'.format(opt.expr_dir, epoch, i))
            if logger:
                logger.info("Model saved to {}/netCRNN_{}_{}.pth".format(opt.expr_dir, epoch, i))
            else:
                print("Model saved to {}/netCRNN_{}_{}.pth".format(opt.expr_dir, epoch, i))
        
# Trainer function for pruning
def trainer(model, criterion, optimizer, epoch, callback):
    return train(model, criterion_c, optimizer, epoch, opt, callback)
        
if __name__=='__main__':
    ''' 
    # Parse the arguments
    # Some useful parameters:
    # batchSize: Set the size of batch
    # expr_dir: Set saved directory path
    # valInterval: Set how many intervals after we do evaluation
    # saveInterval: Set how many intervals after we save the model
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainRoot', required=True, help='path to dataset')
    parser.add_argument('--valRoot', required=True, help='path to dataset')
    # Set pretrain or finetune
    parser.add_argument('--ft', action='store_true', help='set finetune as true means use sample dataset to finetune the model')
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
    parser.add_argument('--expr_dir', default='/data/yunhe/nni_crnn_finetune_pruned_l1_multi3_15', help='Where to store samples and models')
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
    parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
    # Argument for pruning
    parser.add_argument('--prune', action='store_true', help='Whether to do pruning')
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
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batchSize,
        shuffle=True, num_workers=int(opt.workers),
        collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

    nclass = len(opt.alphabet) + 1
    nc = 1

    converter = utils.strLabelConverter(opt.alphabet)
    criterion = CTCLoss()
    
    if opt.finetune:
        crnn = CRNN_p(opt.imgH, nc, nclass, opt.nh)
    else:
        crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
    crnn.apply(weights_init)

    if opt.pretrained != '':
        if logger:
            logger.info('loading pretrained model from %s' % opt.pretrained)
        else:
            logger.info('loading pretrained model from %s' % opt.pretrained)
        
        if not opt.finetune:
            crnn.load_state_dict(load_multi(opt.pretrained), strict=True)
        else:
            crnn.load_state_dict(torch.load(opt.pretrained), strict=True)
    
    logger.info(crnn)

    image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
    text = torch.IntTensor(opt.batchSize * 5)
    length = torch.IntTensor(opt.batchSize)

    if opt.cuda:
        crnn.cuda()
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
        image = image.cuda()
        criterion = criterion.cuda()

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                               betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(crnn.parameters())
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

    logger.info('start training!')
    
    # Pruning setting if opt.prune is set
    if opt.prune:
        configure_list = [{'sparsity': 0.5,'op_types': ['Conv2d'],'op_names': ['module.cnn.conv2']},
            {'sparsity': 0.7,'op_types': ['Conv2d'],'op_names': ['module.cnn.conv3', 'module.cnn.conv4']}, 
            {'sparsity': 0.85,'op_types': ['Conv2d'],'op_names': ['module.cnn.conv5', 'module.cnn.conv6']}]
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
            train(crnn, train_loader, criterion, optimizer, epoch, opt, logger, callback=None)
            pruner.export_model(model_path='{}/pruned_fots{}.pth'.format(opt.expr_dir, epoch), \
                                mask_path='{}/mask_fots{}.pth'.format(opt.expr_dir, epoch))
        # Train the model from scratch
        else:
            train(crnn, train_loader, criterion, optimizer, epoch, opt, logger, callback=None)
