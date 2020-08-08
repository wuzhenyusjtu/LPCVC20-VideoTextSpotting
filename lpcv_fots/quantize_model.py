# Code for l1 0.5 
import argparse
import math

import cv2
import numpy as np
import numpy.random as nprnd
import os
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import tqdm

import datasets
from model import FOTSModel
from modules.parse_polys import parse_polys

from collections import OrderedDict
from model_q import qresnet34
from train_sample import load_multi

def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=False))
    return nn.Sequential(*modules)

class Decoder(nn.Module):
    def __init__(self, in_channels, squeeze_channels):
        super().__init__()
        self.squeeze = conv(in_channels, squeeze_channels)
        self.add_relu = torch.nn.quantized.FloatFunctional()
    def forward(self, x, encoder_features):
        x = self.squeeze(x)
        x = F.interpolate(x, size=(encoder_features.shape[2], encoder_features.shape[3]), mode='bilinear', align_corners=True)
        up = self.add_relu.cat([encoder_features, x], 1)
        return up

# class Center(nn.Module):
class FOTSModel_q(nn.Module):
    def __init__(self, crop_height=640):
        super().__init__()
        self.crop_height = crop_height
        self.resnet = qresnet34(pretrained=False)
        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64
        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4
        
        self.center = nn.Sequential(
            conv(256, 256, stride=2),
            conv(256, 512)
        )
        self.decoder4 = Decoder(512, 256)
        self.decoder3 = Decoder(512, 128)
        self.decoder2 = Decoder(256, 64)
        self.decoder1 = Decoder(128, 32)
        self.remove_artifacts = conv(64, 64)
        
        self.confidence = conv(64, 1, kernel_size=1, padding=0, bn=False, relu=False)
        self.distances = conv(64, 4, kernel_size=1, padding=0, bn=False, relu=False)
        self.angle = conv(64, 1, kernel_size=1, padding=0, bn=False, relu=False)
        
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # self.dequant_op = nn.quantized.DeQuantize()
    
        # Code for early exit
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 1)
        self.init_fc(self.fc)
        self.down_conv = conv(64, 64, kernel_size=3, stride=4)
        self.last_conv = nn.Conv2d(64, 64, 1)
        self.sigm = nn.Sigmoid()
        
        self.rejector = nn.Sequential(OrderedDict([
            ('r1', torch.quantization.QuantStub()),
            ('r2', self.last_conv),
            ('r3', nn.MaxPool2d(2, stride=2)),
            ('r4', self.down_conv),
            ('r5', self.avgpool),
            ('r6', nn.Flatten()),
            ('r7', torch.quantization.DeQuantStub()),
        ]))
        
        self.part1 = nn.Sequential(OrderedDict([
            ('r1', torch.quantization.QuantStub()),
            ('r2', self.conv1),
            ('r3', nn.MaxPool2d(2, stride=2)),
            ('r4', self.encoder1),
            ('r5', torch.quantization.DeQuantStub()),
        ]))
        self.part2 = nn.Sequential(OrderedDict([
            ('r1', torch.quantization.QuantStub()),
            ('r2', self.encoder2),
            ('r3', torch.quantization.DeQuantStub()),
        ]))
        self.part3 = Part3(self.encoder3, self.encoder4, self.center, self.decoder4, self.decoder3, self.decoder2, \
                           self.decoder1, self.remove_artifacts,  self.confidence, self.distances, self.angle, self.crop_height)
        
    def init_fc(self, fc):
        torch.nn.init.normal_(fc.weight, mean=0, std=1)
        torch.nn.init.normal_(fc.bias, mean=0, std=1)
    
    def fuse_conv(self, model):
        torch.quantization.fuse_modules(model, ['0', '1', '2'], inplace=True)
        
    def fuse_model(self):
        self.fuse_conv(self.conv1)
        self.resnet.fuse_model()
        for child in self.center.children():
            self.fuse_conv(child)
        self.fuse_conv(self.decoder1.squeeze)
        self.fuse_conv(self.decoder2.squeeze)
        self.fuse_conv(self.decoder3.squeeze)
        self.fuse_conv(self.decoder4.squeeze)
        self.fuse_conv(self.remove_artifacts)
        self.fuse_conv(self.rejector.r4)
        
    def forward(self, x):
        e1 = self.part1(x)
        e2 = self.part2(e1)
        # Code for early exit
        x = self.rejector(e2)
        x = self.fc(x)
        x = self.sigm(x)
        if x < 0.5:
            return None, None, None
        confidence, distances, angle = self.part3(e1, e2)
        return confidence, distances, angle
    
class Part3(nn.Module):
    def __init__(self, encoder3, encoder4, center, decoder4, decoder3, decoder2, decoder1, remove_artifacts, \
                confidence, distances, angle, crop_height):
        super(Part3, self).__init__()
        self.encoder3 = encoder3
        self.encoder4 = encoder4
        self.center = center
        self.decoder4 = decoder4
        self.decoder3 = decoder3
        self.decoder2 = decoder2
        self.decoder1 = decoder1
        self.remove_artifacts = remove_artifacts
        self.confidence = confidence
        self.distances = distances
        self.angle = angle
        self.crop_height = crop_height
        
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, e1, e2):
        e1 = self.quant(e1)
        e2 = self.quant(e2)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        f = self.center(e4)

        d4 = self.decoder4(f, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        final = self.remove_artifacts(d1)

        confidence = self.confidence(final)
        distances = self.distances(final)
        distances = self.dequant(distances)
        final = self.dequant(final)
        distances = torch.sigmoid(distances) * self.crop_height
        final = self.quant(final)
        angle = self.angle(final)
        angle = self.dequant(angle)
        angle = torch.sigmoid(angle) * np.pi / 2
        confidence = self.dequant(confidence)
        return confidence, distances, angle

def evaluate_net(model, images_folder):
    model.eval()
    cnt = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(os.listdir(images_folder)[0:5], desc='Test', ncols=80)
        for image_name in pbar:
            if not image_name.startswith('res'):
                print(image_name)
                continue
            prefix = image_name[:image_name.rfind('.')]
            image = cv2.imread(os.path.join(images_folder, image_name), cv2.IMREAD_COLOR)
            scale_y = 270 / image.shape[0]  # 1248 # 704
            scaled_image = cv2.resize(image, dsize=(0, 0), fx=scale_y, fy=scale_y, interpolation=cv2.INTER_CUBIC)
            orig_scaled_image = scaled_image.copy()

            scaled_image = scaled_image[:, :, ::-1].astype(np.float32)
            scaled_image = (scaled_image / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            image_tensor = torch.from_numpy(np.expand_dims(np.transpose(scaled_image, axes=(2, 0, 1)), axis=0)).float()

            confidence, distances, angle = net(image_tensor)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrate-folder', type=str, required=True, help='Path to folder with images for calibrating')
    parser.add_argument('--save-dir', type=str, default='./saved_models', help='Path to saved model dir')
    parser.add_argument('--pretrain-model', type=str, default=None, help='Pretrained model name in save dir')
    parser.add_argument('--backends', type=str, default='fbgemm', help='Quantization backends')
    args = parser.parse_args()

    net = FOTSModel_q()
    
    checkpoint_name = args.pretrain_model
    checkpoint = torch.load(checkpoint_name, map_location='cpu')
    
    net.load_state_dict(checkpoint, strict=False)
    _dummy_input_data = torch.rand(1, 3, 299, 299)
    net.eval()
    
    qcf = torch.quantization.get_default_qconfig(args.backends)
    net.rejector.qconfig = qcf
    net.part1.qconfig = qcf
    net.part2.qconfig = qcf
    net.part3.qconfig = qcf
    net.fuse_model()
    torch.quantization.prepare(net.rejector, inplace=True)
    torch.quantization.prepare(net.part1, inplace=True)
    torch.quantization.prepare(net.part2, inplace=True)
    torch.quantization.prepare(net.part3, inplace=True)
    
    images_folder = args.calibrate_folder
    evaluate_net(net, images_folder)
    
    torch.quantization.convert(net.rejector, inplace=True)
    torch.quantization.convert(net.part1, inplace=True)
    torch.quantization.convert(net.part2, inplace=True)
    torch.quantization.convert(net.part3, inplace=True)

    backend = args.backends
    torch.jit.save(torch.jit.script(net.rejector), '{}/rejector_{}.torchscript'.format(args.save_dir, backend))
    torch.jit.save(torch.jit.script(net.part1), '{}/part1_{}.torchscript'.format(args.save_dir, backend))
    torch.jit.save(torch.jit.script(net.part2), '{}/part2_{}.torchscript'.format(args.save_dir, backend))
    torch.jit.save(torch.jit.script(net.part3), '{}/part3_{}.torchscript'.format(args.save_dir, backend))
