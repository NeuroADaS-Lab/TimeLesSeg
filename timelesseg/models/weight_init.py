from torch import nn

from .PlainUNet.residual_conv_blocks import BasicBlockD, BottleneckD
from .MedNeXT.blocks import MedNeXtBlock


class InitWeights_He(object):
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class InitWeights_XavierUniform(object):
    def __init__(self, gain: int = 1):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.xavier_uniform_(module.weight, self.gain)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


def init_last_bn_before_add_to_0(module):
    if isinstance(module, BasicBlockD):
        module.conv2.norm.weight = nn.init.constant_(module.conv2.norm.weight, 0)
        module.conv2.norm.bias = nn.init.constant_(module.conv2.norm.bias, 0)
    if isinstance(module, BottleneckD):
        module.conv3.norm.weight = nn.init.constant_(module.conv3.norm.weight, 0)
        module.conv3.norm.bias = nn.init.constant_(module.conv3.norm.bias, 0)
    if isinstance(module, MedNeXtBlock):
        module.norm2.weight = nn.init.constant_(module.norm2.weight, 0)
        module.norm2.bias = nn.init.constant_(module.norm2.bias, 0)
