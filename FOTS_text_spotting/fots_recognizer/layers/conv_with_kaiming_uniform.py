from torch import nn

from detectron2.layers import Conv2d


def conv_with_kaiming_uniform():
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation * (kernel_size - 1) // 2, dilation=dilation, groups=1, bias=False)
        nn.init.kaiming_uniform_(conv.weight, a=1)
        module = [conv,]
        module.append(nn.BatchNorm2d(out_channels))
        module.append(nn.ReLU(inplace=True))
        return nn.Sequential(*module)

    return make_conv
