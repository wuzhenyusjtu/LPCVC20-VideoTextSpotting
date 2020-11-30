# Import and settings
from __future__ import print_function
from __future__ import division

import argparse
import random
import torch.utils.data
import torch.nn as nn
import numpy as np
import os
from utils import misc
import dataset

from models.crnn import CRNN
from utils.misc import load_multi

import copy

# Functions for mask pruned model
def get_out_channel(conv):
    weight_tensor = conv.weight
    out_list = []
    for i in range(weight_tensor.shape[0]):
        if sum(sum(sum(weight_tensor[i]))) != 0:
            out_list.append(i)
    return out_list

def set_weight_conv(last_c, out_c, ori_conv):
    bias = ori_conv.bias!=None
    conv = nn.Conv2d(len(last_c), len(out_c), kernel_size=ori_conv.kernel_size, \
                     stride=ori_conv.stride, padding=ori_conv.padding, bias=bias)
    conv_shape = conv.weight.shape
    for o in range(conv_shape[0]):
        for i in range(conv_shape[1]):
            conv.weight[o][i] = ori_conv.weight[out_c[o]][last_c[i]]
    if bias:
        for o in range(conv_shape[0]):
            conv.bias[o] = ori_conv.bias[out_c[o]]
    return conv

def set_conv_bn(last_c, out_c, ori_conv, ori_bn):
    conv = set_weight_conv(last_c, out_c, ori_conv)
    bn = nn.BatchNorm2d(ori_bn.num_features, eps=ori_bn.eps, momentum=ori_bn.momentum, affine=ori_bn.affine, \
                        track_running_stats=ori_bn.track_running_stats)
    bn.num_batches_tracked = ori_bn.num_batches_tracked
    for o in range(conv.weight.shape[0]):
        bn.weight[o] = ori_bn.weight[out_c[o]]
        bn.bias[o] = ori_bn.bias[out_c[o]]
        bn.running_mean[o] = ori_bn.running_mean[out_c[o]]
        bn.running_var[o] = ori_bn.running_var[out_c[o]]
    return conv, bn

def set_bn(out_c, ori_bn):
    bn = nn.BatchNorm2d(len(out_c), eps=ori_bn.eps, momentum=ori_bn.momentum, affine=ori_bn.affine, \
                        track_running_stats=ori_bn.track_running_stats)
    bn.num_batches_tracked = ori_bn.num_batches_tracked
    for o in range(len(out_c)):
        bn.weight[o] = ori_bn.weight[out_c[o]]
        bn.bias[o] = ori_bn.bias[out_c[o]]
        bn.running_mean[o] = ori_bn.running_mean[out_c[o]]
        bn.running_var[o] = ori_bn.running_var[out_c[o]]
    return bn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    # TODO(meijieru): epoch -> iter
    parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
    parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
    parser.add_argument('--expr_dir', default='nni_models', help='Where to store samples and models')
    parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
    parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
    opt = parser.parse_args()
    opt.cuda = True
    opt.adadelta = True
    # opt.pretrained = '/home/yunhexue/nni_crnn_results/base_train_adadelta/netCRNN_0_0.pth'
    print(opt)

    # Get converter, transformer, 
    if not os.path.exists(opt.expr_dir):
        os.makedirs(opt.expr_dir)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    transformer = dataset.resizeNormalize((100, 32))

    nclass = len(opt.alphabet) + 1
    nc = 1

    converter = misc.strLabelConverter(opt.alphabet)

    crnn = CRNN(opt.imgH, nc, nclass, opt.nh)
    if opt.pretrained != '':
        print('loading pretrained model from %s' % opt.pretrained)
        crnn.load_state_dict(load_multi(opt.pretrained), strict=False)

    # Process pruned conv2d and batchnorm2d, store them in a dictionary
    crnn_l = list(crnn.cnn._modules.items())

    last_channels = [0]

    crnn_new = CRNN(opt.imgH, nc, nclass, opt.nh)
    crnn_new = copy.deepcopy(crnn)
    new_dict = {}

    for i in range(len(crnn_l)):
        module = crnn_l[i][1]
        if isinstance(module, torch.nn.Conv2d):
            out_channels = get_out_channel(module)
            new = set_weight_conv(last_channels, out_channels, module)
            last_channels = out_channels
            new_dict[crnn_l[i][0]] = new
        if isinstance(module, torch.nn.BatchNorm2d):
            new = set_bn(out_channels, module)
            new_dict[crnn_l[i][0]] = new

    # Process first lstm part
    out_channels = get_out_channel(crnn_new.cnn.conv6)
    for idx, m in enumerate(crnn_new.rnn.children()):
        for j, n in enumerate(m.children()):
            ori_rnn = n
            new_weight_ih_l0 = torch.FloatTensor(n.weight_ih_l0.shape[0], len(out_channels))

            for l in range(len(out_channels)):
                new_weight_ih_l0[:, l] = n.weight_ih_l0[:, out_channels[l]]

            new_weight_ih_l0_r = torch.FloatTensor(n.weight_ih_l0_reverse.shape[0], len(out_channels))
            for l in range(len(out_channels)):
                new_weight_ih_l0_r[:, l] = n.weight_ih_l0_reverse[:, out_channels[l]]

            break
        break
    
    # Copy all lstm data into new one
    test_rnn = nn.LSTM(len(out_channels), 256, bidirectional=True)
    test_rnn.weight_ih_l0_reverse = nn.Parameter(new_weight_ih_l0_r)
    test_rnn.weight_ih_l0 = nn.Parameter(new_weight_ih_l0)

    test_rnn.weight_hh_l0 = ori_rnn.weight_hh_l0
    test_rnn.bias_ih_l0 = ori_rnn.bias_ih_l0
    test_rnn.bias_hh_l0 = ori_rnn.bias_hh_l0
    test_rnn.weight_hh_l0_reverse = ori_rnn.weight_hh_l0_reverse
    test_rnn.bias_ih_l0_reverse = ori_rnn.bias_ih_l0_reverse
    test_rnn.bias_hh_l0_reverse = ori_rnn.bias_hh_l0_reverse

    # Get new crnn model
    crnn_new.cnn.conv1 = new_dict['conv1']
    crnn_new.cnn.conv2 = new_dict['conv2']
    crnn_new.cnn.conv3 = new_dict['conv3']
    crnn_new.cnn.conv4 = new_dict['conv4']
    crnn_new.cnn.conv5 = new_dict['conv5']
    crnn_new.cnn.conv6 = new_dict['conv6']
    crnn_new.cnn.batchnorm2 = new_dict['batchnorm2']
    crnn_new.cnn.batchnorm4 = new_dict['batchnorm4']
    crnn_new.cnn.batchnorm6 = new_dict['batchnorm6']
    for idx, m in enumerate(crnn_new.rnn.children()):
        m.rnn = test_rnn
        break
    print(crnn_new)

    torch.save(crnn_new.state_dict(), '{}/prune_export_CRNN.pt'.format(opt.expr_dir))
    print('CRNN pruned model saved to {}/prune_export_CRNN.pt'.format(opt.expr_dir))
