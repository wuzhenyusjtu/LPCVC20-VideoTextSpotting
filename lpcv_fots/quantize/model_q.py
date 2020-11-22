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
    return _qresnet('resnet34', QuantizableBasicBlock, [3,4,6,3], pretrained, progress, quantize, **kwargs)
def _qresnet(arch, block, layers, pretrained, progress, quantize, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url('https//downloads.pytorch.org/models/resnet34-333f7ec4.pth',
                                                       progress=progress))
        print("Load model successfully")
    
    return model    
