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

import sys 
sys.path.append("..") 
from prune.model_pruned import conv3x3, conv1x1, resnet34, _resnet, BasicBlock, _ResNet

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
        torch.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
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