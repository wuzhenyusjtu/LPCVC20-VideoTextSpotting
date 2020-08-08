# l1 0.5 pruner
# Quantize l1 0.5 fots model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import torch.nn.functional as F
import torchvision
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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                    groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3,4,6,3], pretrained, progress, **kwargs)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = _ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        
        return out

class _ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be none")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.nb3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, 
                           previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                               dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def forward(self, x):
        return self._forward_impl(x)

def _replace_relu(module):
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)
    for key, value in reassign.items():
        module._modules[key] = value 
        
def quantize_model(model, backend):
    _dummy_input_data = torch.rand(1, 3, 224, 224)
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeErrir("Quantized backend not supported")
    torch.backends.quantized.engine = backend
    model.eval()
    
    if backend == 'fbgemm':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_per_channel_weight_observer)
    elif backend == 'qnnpack':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_weight_observer)
    model.fuse_model()
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    
    return 
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

# class QuantizableResNet(ResNet):
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
        fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
        for m in self.modules():
            if type(m) == QuantizableBasicBlock:
                m.fuse_model()
def qresnet34(pretrained=False, progress=True, quantize=True, **kwargs):
    return _qresnet('resnet34', QuantizableBasicBlock, [3,4,6,3], pretrained, progress, quantize, **kwargs)
def _qresnet(arch, block, layers, pretrained, progress, quantize, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url('https//downloads.pytorch.org/models/resnet34-333f7ec4.pth',
                                                       progress=progress))
        print("Load model successfully")
    
    return model    

def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


class Decoder(nn.Module):
    def __init__(self, in_channels, squeeze_channels):
        super().__init__()
        self.squeeze = conv(in_channels, squeeze_channels)

    def forward(self, x, encoder_features):
        x = self.squeeze(x)
        x = F.interpolate(x, size=(encoder_features.shape[2], encoder_features.shape[3]),
                          mode='bilinear', align_corners=True)
        up = torch.cat([encoder_features, x], 1)
        return up


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
        self.encoder4 = self.resnet.layer4  # 512

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

    def forward(self, x):
        
        x = self.quant(x)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
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
        distances = torch.sigmoid(distances) * self.crop_height
        angle = self.angle(final)
        angle = self.dequant(angle)
        angle = torch.sigmoid(angle) * np.pi / 2
        confidence = self.dequant(confidence)
        return confidence, distances, angle
    
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
