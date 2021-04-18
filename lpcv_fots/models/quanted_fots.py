# l1 0.5 pruner
# Quantize l1 0.5 fots model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.__version__)
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['resnet34']
model_urls = {
    'resnet34': 'https://downloads.pytorch.org/models/resnet34-333f7ec4.pth'
}

import sys

sys.path.append("..")
from models.pruned_fots import resnet34, BasicBlock, _ResNet
from models.fots import conv, Decoder

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
        # y = torch.isnan(out)
        # assert y[torch.where(y == True)].item(), "weight is {}\n input is{}\n activation is {}".format(
        #         self.conv1.weight, x, out
        # )
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

    return _qresnet('resnet34', QuantizableBasicBlock, [3, 4, 6, 3], pretrained, progress, quantize, **kwargs)


'''
Class for Quantizeable FOTs model. Some details:
- Original FOTs model forward process can be found in FOTS in models.FOTS
- Adding early exit module into FOTs model:
    -- Early exit module uses the results of encoder2 layer as inputs. If the early exit modules judges that the process
    need to go on, then it means that the whole FOTs model needs to go on and the results of both encoder1 and encoder2
    should be passed to encoder3 and the later on layers.
    -- Problem: Quantized model is as a whole. It cannot deal with the if-else branch operations.
    -- Solution: Devide whole model into three parts:
        Part1: Encoder3 and layers before.
        Part2: Encoder4 and layers after.
        So that we can quantize these two parts separately and get intermediate results for early exit module.
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
        )  # 32
        self.encoder1 = self.resnet.layer1  # 32
        self.encoder2 = self.resnet.layer2  # 64
        self.encoder3 = self.resnet.layer3  # 128
        self.encoder4 = self.resnet.layer4  # 256

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

        self.part1 = Part1(self.conv1, self.encoder1, self.encoder2, self.encoder3)

        self.part2 = Part2(self.encoder4, self.center, self.decoder4, self.decoder3, self.decoder2,
                           self.decoder1, self.remove_artifacts, self.confidence, self.distances, self.angle,
                           self.crop_height)

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

    def forward(self, x):
        e1, e2, e3 = self.part1(x)
        confidence, distances, angle = self.part2(e1, e2, e3)
        return confidence, distances, angle


'''
Part1 includes modules until the encoder 3 layer, an out-of-distribution detection will be inserted between part one and
two in implementation.
'''


class Part1(nn.Module):
    def __init__(self, conv1, encoder1, encoder2, encoder3):
        super(Part1, self).__init__()
        self.conv1 = conv1
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e1 = self.dequant(e1)
        e2 = self.dequant(e2)
        e3 = self.dequant(e3)
        return e1, e2, e3


'''
Part2 includes modules from the encoder4 layer to the rest of the FOTs model.
'''


class Part2(nn.Module):
    def __init__(self, encoder4, center, decoder4, decoder3, decoder2, decoder1, remove_artifacts,
                 confidence, distances, angle, crop_height):
        super(Part2, self).__init__()
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

    def forward(self, e1, e2, e3):
        e1 = self.quant(e1)
        e2 = self.quant(e2)
        e3 = self.quant(e3)
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
        distances = torch.sigmoid(distances) * self.crop_height
        angle = self.angle(final)
        angle = self.dequant(angle)
        angle = torch.sigmoid(angle) * np.pi / 2
        confidence = self.dequant(confidence)
        return confidence, distances, angle
