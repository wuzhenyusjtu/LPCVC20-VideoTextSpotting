# l1 0.5 pruner
# Quantize l1 0.5 fots model
import numpy as np
import torch
import torch.nn as nn
print(torch.__version__)
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
__all__ = ['resnet34']
model_urls = {
    'resnet34':'https://downloads.pytorch.org/models/resnet34-333f7ec4.pth'
}
# model_path = '../resnet50-19c8e357.pth'

import sys 
sys.path.append("..") 
from models.pruned_fots import conv3x3, conv1x1, resnet34, _resnet, BasicBlock, _ResNet
from models.fots import conv, Decoder

from collections import OrderedDict


'''
Quantizable Basic Block for ResNet. Some details:
- Replace add operation with torch.nn.quantized.FloatFunctional() to make it quantizable
- Method fuse_model() is used to fuse conv+bn+relu layers into one layer to make it quantizable, otherwise quantized results nonsense.
- More details about why do fuse_model() please check Pytorch offical documents.
'''
class QuantizableBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super(QuantizableBasicBlock, self).__init__(*args, **kwargs)
        # Wrapper class to make stateless float operations stateful, can be replaced with quantized versions
        self.add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.add_relu.add_relu(out, identity)
        
        return out

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu'], ['conv2', 'bn2']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)

'''
Quantizable Resnet constructed by QuantizableBasicBlock. Some details:
- Methods QuantStub/DeQuantStub do pre/after quantization for inputs/outputs.
- Method fuse_model fuse the 'conv1', 'bn1', 'relu' layers in ResNet and call fuse_model functions in its modules to do fuse for all parts.
'''
class ResNet(_ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
        for m in self.modules():
            if type(m) == QuantizableBasicBlock:
                m.fuse_model()

'''
Wrapper functions for quantized models (Consistent with resnet from Pytorch)
'''
def qresnet34(pretrained=False, progress=True, quantize=True, **kwargs):
    def _qresnet(arch, block, layers, pretrained, progress, quantize, **kwargs):
        model = ResNet(block, layers, **kwargs)
        if pretrained:
            model.load_state_dict(load_state_dict_from_url('https//downloads.pytorch.org/models/resnet34-333f7ec4.pth',
                                                           progress=progress))
            print("Load model successfully")

        return model
    return _qresnet('resnet34', QuantizableBasicBlock, [3,4,6,3], pretrained, progress, quantize, **kwargs)


'''
Class for Quantizeable FOTs model. Some details:
- Original FOTs model forward process can be found in FOTS in models.FOTS
- Adding early exit module into FOTs model:
    -- Early exit module uses the results of encoder2 layer as inputs. If the early exit modules judges that the process need to go on, then it means that the whole FOTs model needs to go on and the results of both encoder1 and encoder2 should be passed to encoder3 and the later on layers.
    -- Problem: Quantized model is as a whole. It cannot deal with the if-else branch operations.
    -- Solution: Devide whole model into three parts:
        Part1: Encoder1 and layers before.
        Part2: Encoder2.
        Part3: Encoder3 and layers after.
        Rejector: Early exit module
        So that we can quantize these three parts separately and get intermediate results for early exit module.
- Method fuse_conv（）can fuse conv+bn+relu layers into one layer for future model quantization.
- Method fuse_model() fuse all modules in FOTs model.
'''


class FOTS_quanted(nn.Module):
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

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
                           self.decoder1, self.remove_artifacts, self.confidence, self.distances, self.angle,
                           self.crop_height)

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


'''
All modules come from the encoder3 layer and layers afterwards from the FOTs model.
Integrate all modules into one class called Part3.
'''
class Part3(nn.Module):
    def __init__(self, encoder3, encoder4, center, decoder4, decoder3, decoder2, decoder1, remove_artifacts,
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