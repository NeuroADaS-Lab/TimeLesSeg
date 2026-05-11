import sys
import torch
from torch import nn
import enum

if sys.version_info >= (3, 11):    
    auto = enum.auto
    StrEnum = enum.StrEnum
else:

    auto = None
    class StrEnum(str, enum.Enum):
        pass


class Net(StrEnum):
    if auto is None:
        MEDNEXT = 'mednext'
        UNet = 'unet'
        ResEncUNet = 'resencunet'
        ResUNet = 'resunet'
    else:
        MEDNEXT = auto()
        UNet = auto()
        ResEncUNet = auto()
        ResUNet = auto()



def features_per_stage(start_num_features: int, n_stages: int, max_num_features: int) -> tuple[int, ...]:
    return tuple([min(max_num_features, start_num_features * 2 ** i) for i in range(n_stages)])

def _maybe_convert_scalar_to_list(scalar, n: int) -> list:
    if isinstance(scalar, int):
        scalar = [scalar] * n
    assert len(scalar) == n
    return list(scalar)

def maybe_convert_scalar_to_list(scalar, dim_or_convop) -> list:
    if not isinstance(dim_or_convop, int):
        dim_or_convop = convert_conv_op_to_dim(dim_or_convop)

    return _maybe_convert_scalar_to_list(scalar, dim_or_convop)

def convert_conv_op_to_dim(conv_op) -> int:
    dim = conv_op.__name__[-2]
    return int(dim)

def convert_dim_to_conv_op(dim: int):
    return getattr(nn, f'Conv{dim}d')

def get_matching_convtransp(dim_or_conv_op):
    """
    """
    if not isinstance(dim_or_conv_op, int):
        dim_or_conv_op = convert_conv_op_to_dim(dim_or_conv_op)

    return getattr(nn, f'ConvTranspose{dim_or_conv_op}d')

def get_matching_pool_op(dim_or_convop,
                         adaptive: bool = False,
                         pool_type: str = 'avg'):
    """
    :param dimension:
    :param adaptive:
    :param pool_type: either 'avg' or 'max'
    :return:
    """
    if not isinstance(dim_or_convop, int):
        dim_or_convop = convert_conv_op_to_dim(dim_or_convop)

    start = ''
    if adaptive:
        start += 'Adaptive'
    return getattr(nn, f'{start}{pool_type.capitalize()}Pool{dim_or_convop}d')

def check_two_nets_are_equal(net1, net2, *, kwargs1: dict = None, kwargs2: dict = None, seed: int = 9999):

    torch.manual_seed(seed)
    if kwargs1 is not None:
        net1 = net1(**kwargs1)
        kwargs2 = kwargs2 or kwargs1

    torch.manual_seed(seed)
    if kwargs2 is not None:
        net2 = net2(**kwargs2)

    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def get_default_groupnorm_kwargs(in_channels: int, num_groups: int = None) -> dict:
    """
    :param num_groups: If not provided or None, set to in_channels.
    """
    num_groups = num_groups or in_channels

    # by default use learnable alfa and beta
    groupnorm_kwargs = {'num_channels': in_channels, 'num_groups': num_groups, 'affine': True}
    return groupnorm_kwargs


if __name__ == "__main__":
    net1 = nn.Conv3d
    kwargs = dict(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1, stride = 2)
    net2 = nn.Conv3d
    assert check_two_nets_are_equal(net1, net2, kwargs1=kwargs)

    assert convert_conv_op_to_dim(net1) == 3

    assert maybe_convert_scalar_to_list(3, dim_or_convop=3) == maybe_convert_scalar_to_list(3, nn.Conv3d) == [3, 3, 3]

    assert get_matching_pool_op(dim_or_convop = 3, adaptive=False, pool_type='avg') is \
        get_matching_pool_op(dim_or_convop=nn.Conv3d, adaptive=False, pool_type='avg') is \
        nn.AvgPool3d

    assert convert_dim_to_conv_op(3) is nn.Conv3d

    assert convert_conv_op_to_dim(nn.Conv3d) == convert_conv_op_to_dim(nn.Conv3d) and \
        convert_conv_op_to_dim(nn.Conv2d) == convert_conv_op_to_dim(nn.Conv2d)

    assert get_matching_pool_op(3, adaptive=True, pool_type='max') == get_matching_pool_op(3, adaptive=True, pool_type='max')