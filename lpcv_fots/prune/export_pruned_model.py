# Functions for mask pruned model
import sys 
sys.path.append("..") 

import torch
from utils.train_utils import load_multi
from standard.model import FOTSModel
import argparse

import cv2
import numpy as np
import os
import torch
import torch.utils.data

import copy
import torch.nn as nn

def get_out_channel(conv):
    weight_tensor = conv.weight
    out_list = []
    for i in range(weight_tensor.shape[0]):
        if sum(sum(sum(weight_tensor[i]))) != 0:
            out_list.append(i)
#             print(i)
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

def set_weight_conv_in(last_c, out_c, ori_conv):
    bias = ori_conv.bias!=None
    conv = nn.Conv2d(len(last_c), ori_conv.out_channels, kernel_size=ori_conv.kernel_size, \
                     stride=ori_conv.stride, padding=ori_conv.padding, bias=bias)
    conv_shape = conv.weight.shape
    for o in range(conv_shape[0]):
        for i in range(conv_shape[1]):
            conv.weight[o][i] = ori_conv.weight[o][last_c[i]]
    if bias:
        for o in range(conv_shape[0]):
            conv.bias[o] = ori_conv.bias[o]
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

def set_bn_in(out_c, ori_bn):
    bn = nn.BatchNorm2d(ori_bn.num_features, eps=ori_bn.eps, momentum=ori_bn.momentum, affine=ori_bn.affine, \
                        track_running_stats=ori_bn.track_running_stats)
    bn.num_batches_tracked = ori_bn.num_batches_tracked
    for o in range(ori_bn.num_features):
        bn.weight[o] = ori_bn.weight[o]
        bn.bias[o] = ori_bn.bias[o]
        bn.running_mean[o] = ori_bn.running_mean[o]
        bn.running_var[o] = ori_bn.running_var[o]
    return bn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='./saved_models', help='Path to saved model dir')
    parser.add_argument('--prune-model', type=str, default=None, help='Pretrained model name in save dir')
    args = parser.parse_args()
    
    model = FOTSModel().to(torch.device("cpu"))
    model = model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=32, \
                                                              verbose=True, threshold=0.05, threshold_mode='rel')

    checkpoint_name = args.prune_model

    checkpoint = torch.load(checkpoint_name, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    
    fots_l = list(model.resnet._modules.items())
    
    last_channels = [0, 1, 2]

    model_new = FOTSModel().to(torch.device("cpu"))
    model_new = model.eval()
    model_new = copy.deepcopy(model)
    new_dict = {}

    for i in range(4):
        module = fots_l[i][1]
        if isinstance(module, torch.nn.Conv2d):
            out_channels = get_out_channel(module)
            new = set_weight_conv(last_channels, out_channels, module)
            last_channels = out_channels
            new_dict[fots_l[i][0]] = new
        if isinstance(module, torch.nn.BatchNorm2d):
            new = set_bn(out_channels, module)
            new_dict[fots_l[i][0]] = new
    model_new.resnet.conv1 = new_dict['conv1']
    model_new.resnet.bn1 = new_dict['bn1']

    # Code for layer 1

    fots_l = list(model.resnet.layer1._modules.items())
    new_dict_layer1 = {}
    for i in range(len(fots_l)):
        module = fots_l[i][1]
        new_dict_layer1[i] = {}
        module_l = list(module._modules.items())

        for j in range(len(module_l)):
            module_b = module_l[j][1]
            if isinstance(module_b, torch.nn.Conv2d):
                out_channels = get_out_channel(module_b)
                new = set_weight_conv(last_channels, out_channels, module_b)
                last_channels = out_channels
                new_dict_layer1[i][module_l[j][0]] = new
            if isinstance(module_b, torch.nn.BatchNorm2d):
                new = set_bn(out_channels, module_b)
                new_dict_layer1[i][module_l[j][0]] = new
                
    for idx, m in enumerate(model_new.resnet.layer1.children()):
        conv_dict = new_dict_layer1[idx]
        m.conv1 = conv_dict['conv1']
        m.bn1 = conv_dict['bn1']
        m.conv2 = conv_dict['conv2']
        m.bn2 = conv_dict['bn2']

    # Code for layer 2

    last_down_channels_1 = last_channels
    fots_l = list(model.resnet.layer2._modules.items())
    new_dict_layer2 = {}
    for i in range(len(fots_l)):
        module = fots_l[i][1]
        new_dict_layer2[i] = {}
        module_l = list(module._modules.items())

        for j in range(len(module_l)):
            module_b = module_l[j][1]
            if isinstance(module_b, torch.nn.Conv2d):
                out_channels = get_out_channel(module_b)
                new = set_weight_conv(last_channels, out_channels, module_b)
                last_channels = out_channels
                new_dict_layer2[i][module_l[j][0]] = new
            elif isinstance(module_b, torch.nn.BatchNorm2d):
                new = set_bn(out_channels, module_b)
                new_dict_layer2[i][module_l[j][0]] = new
            elif isinstance(module_b, torch.nn.ReLU):
                continue
            else:
                module_down = list(module_b._modules.items())
                new_dict_layer2['downsample'] = {}
                for k in range(len(module_down)):
                    module_k = module_down[k][1]
                    if isinstance(module_k, torch.nn.Conv2d):
                        out_channels = get_out_channel(module_k)
                        new = set_weight_conv(last_down_channels_1, out_channels, module_k)
                        last_channels = out_channels
                        new_dict_layer2['downsample'][module_down[k][0]] = new
                    elif isinstance(module_k, torch.nn.BatchNorm2d):
                        new = set_bn(out_channels, module_k)
                        new_dict_layer2['downsample'][module_down[k][0]] = new

    for idx, m in enumerate(model_new.resnet.layer2.children()):
        conv_dict = new_dict_layer2[idx]
        m.conv1 = conv_dict['conv1']
        m.bn1 = conv_dict['bn1']
        m.conv2 = conv_dict['conv2']
        m.bn2 = conv_dict['bn2']
        if idx == 0:
            m.downsample = nn.Sequential(
                new_dict_layer2['downsample']['0'],
                new_dict_layer2['downsample']['1'],)
            
    # Code for layer 3

    last_down_channels_2 = last_channels
    fots_l = list(model.resnet.layer3._modules.items())
    new_dict_layer3 = {}
    for i in range(len(fots_l)):
        module = fots_l[i][1]
        new_dict_layer3[i] = {}
        module_l = list(module._modules.items())

        for j in range(len(module_l)):
            module_b = module_l[j][1]
            if isinstance(module_b, torch.nn.Conv2d):
                out_channels = get_out_channel(module_b)
                new = set_weight_conv(last_channels, out_channels, module_b)
                last_channels = out_channels
                new_dict_layer3[i][module_l[j][0]] = new
            elif isinstance(module_b, torch.nn.BatchNorm2d):
                new = set_bn(out_channels, module_b)
                new_dict_layer3[i][module_l[j][0]] = new
            elif isinstance(module_b, torch.nn.ReLU):
                continue
            else:
                module_down = list(module_b._modules.items())
                new_dict_layer3['downsample'] = {}
                for k in range(len(module_down)):
                    module_k = module_down[k][1]
                    if isinstance(module_k, torch.nn.Conv2d):
                        out_channels = get_out_channel(module_k)
                        new = set_weight_conv(last_down_channels_2, out_channels, module_k)
                        last_channels = out_channels
                        new_dict_layer3['downsample'][module_down[k][0]] = new
                    elif isinstance(module_k, torch.nn.BatchNorm2d):
                        new = set_bn(out_channels, module_k)
                        new_dict_layer3['downsample'][module_down[k][0]] = new


    for idx, m in enumerate(model_new.resnet.layer3.children()):
        conv_dict = new_dict_layer3[idx]
        m.conv1 = conv_dict['conv1']
        m.bn1 = conv_dict['bn1']
        m.conv2 = conv_dict['conv2']
        m.bn2 = conv_dict['bn2']
        if idx == 0:
            m.downsample = nn.Sequential(
                new_dict_layer3['downsample']['0'],
                new_dict_layer3['downsample']['1'],)

    # Code for layer 4

    last_down_channels_3 = last_channels
    fots_l = list(model.resnet.layer4._modules.items())
    new_dict_layer4 = {}
    for i in range(len(fots_l)):
        module = fots_l[i][1]
        new_dict_layer4[i] = {}
        module_l = list(module._modules.items())

        for j in range(len(module_l)):
            module_b = module_l[j][1]
            if isinstance(module_b, torch.nn.Conv2d):
                out_channels = get_out_channel(module_b)
                new = set_weight_conv(last_channels, out_channels, module_b)
                last_channels = out_channels
                new_dict_layer4[i][module_l[j][0]] = new
            elif isinstance(module_b, torch.nn.BatchNorm2d):
                new = set_bn(out_channels, module_b)
                new_dict_layer4[i][module_l[j][0]] = new
            elif isinstance(module_b, torch.nn.ReLU):
                continue
            else:
                module_down = list(module_b._modules.items())
                new_dict_layer4['downsample'] = {}
                for k in range(len(module_down)):
                    module_k = module_down[k][1]
                    if isinstance(module_k, torch.nn.Conv2d):
                        out_channels = get_out_channel(module_k)
                        new = set_weight_conv(last_down_channels_3, out_channels, module_k)
                        last_channels = out_channels
                        new_dict_layer4['downsample'][module_down[k][0]] = new
                    elif isinstance(module_k, torch.nn.BatchNorm2d):
                        new = set_bn(out_channels, module_k)
                        new_dict_layer4['downsample'][module_down[k][0]] = new

    for idx, m in enumerate(model_new.resnet.layer4.children()):
        conv_dict = new_dict_layer4[idx]
        m.conv1 = conv_dict['conv1']
        m.bn1 = conv_dict['bn1']
        m.conv2 = conv_dict['conv2']
        m.bn2 = conv_dict['bn2']
        if idx == 0:
            m.downsample = nn.Sequential(
                new_dict_layer4['downsample']['0'],
                new_dict_layer4['downsample']['1'],)

    last_down_channels_4 = last_channels
    
    # Code for center

    last_channels = last_down_channels_4
    fots_l = list(model.center._modules.items())
    new_dict_center = {}
    for i in range(len(fots_l)):
        module = fots_l[i][1]
        new_dict_center[i] = {}
        module_l = list(module._modules.items())

        for j in range(len(module_l)):
            module_b = module_l[j][1]
            if isinstance(module_b, torch.nn.Conv2d):
                out_channels = get_out_channel(module_b)
                new = set_weight_conv(last_channels, out_channels, module_b)
                last_channels = out_channels
                new_dict_center[i][module_l[j][0]] = new
            elif isinstance(module_b, torch.nn.BatchNorm2d):
                new = set_bn(out_channels, module_b)
                new_dict_center[i][module_l[j][0]] = new
            elif isinstance(module_b, torch.nn.ReLU):
                new_dict_center[i][module_l[j][0]] = module_b

    model_new.center = nn.Sequential(
            nn.Sequential(
                new_dict_center[0]['0'],
                new_dict_center[0]['1'],
                new_dict_center[0]['2'],
            ),
            nn.Sequential(
                new_dict_center[1]['0'],
                new_dict_center[1]['1'],
                new_dict_center[1]['2'],
            )
        )
    last_center_channels = last_channels
    
    # Code for decoder 4

    last_channels = last_center_channels
    fots_l = list(model.decoder4.squeeze._modules.items())
    new_dict_decoder4 = {}
    for i in range(len(fots_l)):
        module_b = fots_l[i][1]
        if isinstance(module_b, torch.nn.Conv2d):
            out_channels = get_out_channel(module_b)
            new = set_weight_conv(last_channels, out_channels, module_b)
            last_channels = out_channels
            new_dict_decoder4[fots_l[i][0]] = new
        elif isinstance(module_b, torch.nn.BatchNorm2d):
            new = set_bn(out_channels, module_b)
            new_dict_decoder4[fots_l[i][0]] = new
        elif isinstance(module_b, torch.nn.ReLU):
            new_dict_decoder4[fots_l[i][0]] = module_b

    model_new.decoder4.squeeze = nn.Sequential(
                new_dict_decoder4['0'],
                new_dict_decoder4['1'],
                new_dict_decoder4['2'],)
    last_decoder4_channels = last_channels
    
    # Code for decoder 3

    last_e4_channels = last_down_channels_4
    last_d4_channels = last_decoder4_channels
    last_channels = []
    for i in range(len(last_e4_channels)):
        last_channels.append(last_e4_channels[i])
        last_channels.append(last_d4_channels[i] + 512)

    fots_l = list(model.decoder3.squeeze._modules.items())
    new_dict_decoder3 = {}
    for i in range(len(fots_l)):
        module_b = fots_l[i][1]
        if isinstance(module_b, torch.nn.Conv2d):
            out_channels = get_out_channel(module_b)
            new = set_weight_conv(last_channels, out_channels, module_b)
            last_channels = out_channels
            new_dict_decoder3[fots_l[i][0]] = new
        elif isinstance(module_b, torch.nn.BatchNorm2d):
            new = set_bn(out_channels, module_b)
            new_dict_decoder3[fots_l[i][0]] = new
        elif isinstance(module_b, torch.nn.ReLU):
            new_dict_decoder3[fots_l[i][0]] = module_b

    model_new.decoder3.squeeze = nn.Sequential(
                new_dict_decoder3['0'],
                new_dict_decoder3['1'],
                new_dict_decoder3['2'],
        )
    last_decoder3_channels = last_channels
    
    # Code for decoder 2

    last_e3_channels = last_down_channels_3
    last_d3_channels = last_decoder3_channels
    last_channels = []
    for i in range(len(last_e3_channels)):
        last_channels.append(last_e3_channels[i])
        last_channels.append(last_d3_channels[i] + 256)

    fots_l = list(model.decoder2.squeeze._modules.items())
    new_dict_decoder2 = {}
    for i in range(len(fots_l)):
        module_b = fots_l[i][1]
        if isinstance(module_b, torch.nn.Conv2d):
            out_channels = get_out_channel(module_b)
            new = set_weight_conv(last_channels, out_channels, module_b)
            last_channels = out_channels
            new_dict_decoder2[fots_l[i][0]] = new
        elif isinstance(module_b, torch.nn.BatchNorm2d):
            new = set_bn(out_channels, module_b)
            new_dict_decoder2[fots_l[i][0]] = new
        elif isinstance(module_b, torch.nn.ReLU):
            new_dict_decoder2[fots_l[i][0]] = module_b

    model_new.decoder2.squeeze = nn.Sequential(
                new_dict_decoder2['0'],
                new_dict_decoder2['1'],
                new_dict_decoder2['2'],
        )
    last_decoder2_channels = last_channels
    
    # Code for decoder 1

    last_e2_channels = last_down_channels_2
    last_d2_channels = last_decoder2_channels
    last_channels = []
    for i in range(len(last_e2_channels)):
        last_channels.append(last_e2_channels[i])
        last_channels.append(last_d2_channels[i] + 128)

    fots_l = list(model.decoder1.squeeze._modules.items())
    new_dict_decoder1 = {}
    for i in range(len(fots_l)):
        module_b = fots_l[i][1]
        if isinstance(module_b, torch.nn.Conv2d):
            out_channels = get_out_channel(module_b)
            new = set_weight_conv(last_channels, out_channels, module_b)
            last_channels = out_channels
            new_dict_decoder1[fots_l[i][0]] = new
        elif isinstance(module_b, torch.nn.BatchNorm2d):
            new = set_bn(out_channels, module_b)
            new_dict_decoder1[fots_l[i][0]] = new
        elif isinstance(module_b, torch.nn.ReLU):
            new_dict_decoder1[fots_l[i][0]] = module_b

    model_new.decoder1.squeeze = nn.Sequential(
                new_dict_decoder1['0'],
                new_dict_decoder1['1'],
                new_dict_decoder1['2'],
        )
    last_decoder1_channels = last_channels
    
    # Code for remove artifacts

    last_e1_channels = last_down_channels_1
    last_d1_channels = last_decoder1_channels
    last_channels = []
    for i in range(len(last_e1_channels)):
        last_channels.append(last_e1_channels[i])
        last_channels.append(last_d1_channels[i] + 64)

    fots_l = list(model.remove_artifacts._modules.items())
    new_dict_art = {}
    for i in range(len(fots_l)):
        module_b = fots_l[i][1]
        if isinstance(module_b, torch.nn.Conv2d):
            out_channels = get_out_channel(module_b)
            new = set_weight_conv_in(last_channels, out_channels, module_b)
            last_channels = out_channels
            new_dict_art[fots_l[i][0]] = new
        elif isinstance(module_b, torch.nn.BatchNorm2d):
            new = set_bn_in(out_channels, module_b)
            new_dict_art[fots_l[i][0]] = new
        elif isinstance(module_b, torch.nn.ReLU):
            new_dict_art[fots_l[i][0]] = module_b

    model_new.remove_artifacts = nn.Sequential(
                new_dict_art['0'],
                new_dict_art['1'],
                new_dict_art['2'],
        )
    last_art_channels = last_channels
    
    # Code for final three convs
    out_channels = get_out_channel(model_new.confidence._modules['0'])
    new_c = set_weight_conv(last_art_channels, out_channels, model_new.confidence._modules['0'])
    model_new.confidence = nn.Sequential(new_c)
    out_channels = get_out_channel(model_new.distances._modules['0'])
    new_d = set_weight_conv(last_art_channels, out_channels, model_new.distances._modules['0'])
    model_new.distances = nn.Sequential(new_d)
    out_channels = get_out_channel(model_new.angle._modules['0'])
    new_a = set_weight_conv(last_art_channels, out_channels, model_new.angle._modules['0'])
    model_new.angle = nn.Sequential(new_a)
    
    model_new.conv1 = nn.Sequential(
            model_new.resnet.conv1,
            model_new.resnet.bn1,
            model_new.resnet.relu,
        )  # 64
    print(model_new)
    
    torch.save(model_new.state_dict(), '{}/pruned_model.pth'.format(args.save_dir))
