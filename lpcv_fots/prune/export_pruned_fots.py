# Functions for mask pruned model
import sys 
sys.path.append("..") 


from models.fots import FOTS
import argparse

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
    
    model = FOTS().to(torch.device("cpu"))
    model = model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=32, \
                                                              verbose=True, threshold=0.05, threshold_mode='rel')

    checkpoint_name = args.prune_model

    checkpoint = torch.load(checkpoint_name, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    
    fots_l = list(model.resnet._modules.items())
    
    last_channels = [0, 1, 2]

    model_new = FOTS().to(torch.device("cpu"))
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

    # Code for Conv1

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

    # Code for Encoders
    def copy_res_layer(last_channels, fots_l):
        last_down_channels = last_channels
        new_dict = {}
        for i in range(len(fots_l)):
            module = fots_l[i][1]
            new_dict[i] = {}
            module_l = list(module._modules.items())

            for j in range(len(module_l)):
                module_b = module_l[j][1]
                if isinstance(module_b, torch.nn.Conv2d):
                    out_channels = get_out_channel(module_b)
                    new = set_weight_conv(last_channels, out_channels, module_b)
                    last_channels = out_channels
                    new_dict[i][module_l[j][0]] = new
                elif isinstance(module_b, torch.nn.BatchNorm2d):
                    new = set_bn(out_channels, module_b)
                    new_dict[i][module_l[j][0]] = new
                elif isinstance(module_b, torch.nn.ReLU):
                    continue
                else:
                    module_down = list(module_b._modules.items())
                    new_dict['downsample'] = {}
                    for k in range(len(module_down)):
                        module_k = module_down[k][1]
                        if isinstance(module_k, torch.nn.Conv2d):
                            out_channels = get_out_channel(module_k)
                            new = set_weight_conv(last_down_channels, out_channels, module_k)
                            last_channels = out_channels
                            new_dict['downsample'][module_down[k][0]] = new
                        elif isinstance(module_k, torch.nn.BatchNorm2d):
                            new = set_bn(out_channels, module_k)
                            new_dict['downsample'][module_down[k][0]] = new
        return last_channels, new_dict
    
    def set_values_res_layer(layer, new_dict):
        for idx, m in enumerate(layer.children()):
            conv_dict = new_dict[idx]
            m.conv1 = conv_dict['conv1']
            m.bn1 = conv_dict['bn1']
            m.conv2 = conv_dict['conv2']
            m.bn2 = conv_dict['bn2']
            if idx == 0:
                m.downsample = nn.Sequential(
                    new_dict['downsample']['0'],
                    new_dict['downsample']['1'],)
                
    last_down_channels_1 = last_channels
    last_channels, new_dict = copy_res_layer(last_channels, list(model.resnet.layer2._modules.items()))
    set_values_res_layer(model_new.resnet.layer2, new_dict)
    
    last_down_channels_2 = last_channels
    last_channels, new_dict = copy_res_layer(last_channels, list(model.resnet.layer3._modules.items()))
    set_values_res_layer(model_new.resnet.layer3, new_dict)
    
    last_down_channels_3 = last_channels
    last_channels, new_dict = copy_res_layer(last_channels, list(model.resnet.layer4._modules.items()))
    set_values_res_layer(model_new.resnet.layer4, new_dict)
    
    last_down_channels_4 = last_channels
    
    # Code for center
    
    last_down_channels_4 = last_channels
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
    
    # Code for decoders
    def copy_decoder(last_channels, fots_l):
        new_dict = {}
        for i in range(len(fots_l)):
            module_b = fots_l[i][1]
            if isinstance(module_b, torch.nn.Conv2d):
                out_channels = get_out_channel(module_b)
                new = set_weight_conv(last_channels, out_channels, module_b)
                last_channels = out_channels
                new_dict[fots_l[i][0]] = new
            elif isinstance(module_b, torch.nn.BatchNorm2d):
                new = set_bn(out_channels, module_b)
                new_dict[fots_l[i][0]] = new
            elif isinstance(module_b, torch.nn.ReLU):
                new_dict[fots_l[i][0]] = module_b
        return last_channels, new_dict
    
    def process_decoder(old_layer, target_layer, last_channels):
        try:
            last_channels, new_dict = copy_decoder(last_channels, list(old_layer.squeeze._modules.items()))
            target_layer.squeeze = nn.Sequential(
                    new_dict['0'],
                    new_dict['1'],
                    new_dict['2'],)
        except:
            last_channels, new_dict = copy_decoder(last_channels, list(old_layer._modules.items()))
            target_layer = nn.Sequential(
                    new_dict['0'],
                    new_dict['1'],
                    new_dict['2'],)
        return last_channels
    
    def get_merged_channels(d_channels, e_channels):
        last_channels = []
        for i in range(len(e_channels)):
            last_channels.append(e_channels[i])
            last_channels.append(d_channels[i] + len(e_channels) * 2)
        return last_channels
            
    last_d4_channels = process_decoder(model.decoder4, model_new.decoder4, last_channels)
    
    last_channels = get_merged_channels(last_d4_channels, last_down_channels_4)
    last_d3_channels = process_decoder(model.decoder3, model_new.decoder3, last_channels)
    
    last_channels = get_merged_channels(last_d3_channels, last_down_channels_3)
    last_d2_channels = process_decoder(model.decoder2, model_new.decoder2, last_channels)
    
    last_channels = get_merged_channels(last_d2_channels, last_down_channels_2)
    last_d1_channels = process_decoder(model.decoder1, model_new.decoder1, last_channels)
    
    last_channels = get_merged_channels(last_d1_channels, last_down_channels_1)
    last_art_channels = process_decoder(model.remove_artifacts, model_new.remove_artifacts, last_channels)
    
    # Code for final three convs
    def copy_last_convs(old_layer, target_layer, last_art_channels):
        out_channels = get_out_channel(old_layer._modules['0'])
        new_c = set_weight_conv(last_art_channels, out_channels, old_layer._modules['0'])
        target_layer = nn.Sequential(new_c)
    
    
    copy_last_convs(model.confidence, model_new.confidence, last_art_channels)
    copy_last_convs(model.distances, model_new.distances, last_art_channels)
    copy_last_convs(model.angle, model_new.angle, last_art_channels)
    
    model_new.conv1 = nn.Sequential(
            model_new.resnet.conv1,
            model_new.resnet.bn1,
            model_new.resnet.relu,
        )  # 64
    print(model_new)
    
    torch.save(model_new.state_dict(), '{}/pruned_model.pth'.format(args.save_dir))
