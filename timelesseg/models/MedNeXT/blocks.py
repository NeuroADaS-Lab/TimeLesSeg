from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timelesseg.models.utils import (
    convert_conv_op_to_dim,
    maybe_convert_scalar_to_list,
    get_default_groupnorm_kwargs
)
from timelesseg.typing import ToIterableInt

class MedNeXtBlock(nn.Module):

    def __init__(
        self,
        conv_op,
        in_channels: int,
        out_channels: int,
        initial_stride: int,
        initial_kernel_size: int,
        r: int = 4,
        norm_op = None,
        norm_op_kwargs = None,
        dropout = None,
        dropout_kwargs = None,
        nonlin = None,
        nonlin_kwargs = None,
        residual: bool = True,
        hidden_dim: int = None,
        perform_second_nonlin: bool = True,
        conv_bias: bool = False
    ):
        """
        Parameters
        -------
        r: expansion ratio as in Swin Transformers
        """

        super().__init__()

        self.input_channels = in_channels
        self.output_channels = out_channels
        self.expansion_ratio = r
        self.residual = residual
        self.perform_second_nonlin = perform_second_nonlin
        initial_stride = maybe_convert_scalar_to_list(initial_stride, conv_op)
        self.stride = initial_stride

        if norm_op_kwargs is None:
            assert norm_op is nn.GroupNorm

        # this is for when in_channels and hidden dim differ
        # e.g., in the first layer, where in_channels is 1 and we want
        # to increment it to N
        hidden_dim = hidden_dim or in_channels
        self.hidden_dim = hidden_dim

        # Depthwise Convolution Layer
        self.conv1 = conv_op(
            in_channels = in_channels,
            out_channels = hidden_dim,
            kernel_size = initial_kernel_size,
            stride = initial_stride,
            padding = np.array(initial_kernel_size) // 2,
            groups = in_channels,
            bias = conv_bias
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_op is not None:
            norm_op_kwargs1 = norm_op_kwargs or get_default_groupnorm_kwargs(hidden_dim, hidden_dim)
            self.norm = norm_op(**norm_op_kwargs1)

        # Expansion Layer with C * R out channels (R == expansion_ratio)
        self.conv2 = conv_op(
            in_channels = hidden_dim,
            out_channels = r * hidden_dim,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = conv_bias
        )

        # GeLU activations default
        if nonlin is not None:
            nonlin_kwargs = nonlin_kwargs or {}
            self.nonlin = nonlin(**nonlin_kwargs)

        # Compression Layer
        self.conv3 = conv_op(
            in_channels = r * hidden_dim,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = conv_bias
        )

        if norm_op is not None:
            norm_op_kwargs2 = norm_op_kwargs or get_default_groupnorm_kwargs(out_channels, out_channels)
            self.norm2 = norm_op(**norm_op_kwargs2)

        # if the number of channels or downsampling has ocurred, a projection is required for residual connection
        if residual and (in_channels != out_channels or any(i != 1 for i in initial_stride)):
            skip_ops = []

            skip_ops.append(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=initial_stride, bias=False)) # bias=False to reproduce Resenc blocks
            if norm_op is not None:
                skip_ops.append(norm_op(**norm_op_kwargs2))

            self.skip = nn.Sequential(*skip_ops)

        else:
            self.skip = nn.Identity()

    def forward(self, x):

        residual = self.skip(x)

        x = self.conv1(x)
        x = self.nonlin(self.conv2(self.norm(x)))

        # normalize before adding residual and f(x)
        x = self.norm2(self.conv3(x))

        if self.residual:
            x += residual

        # final nonlinearity (default true)
        if self.perform_second_nonlin:
            x = self.nonlin(x)

        return x

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"

        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
        # conv1
        output_size_conv1 = np.prod([self.hidden_dim, *size_after_stride], dtype=np.int64)
        # conv2
        output_size_conv2 = np.prod([self.expansion_ratio * self.hidden_dim, *size_after_stride], dtype=np.int64)
        # conv3
        output_size_conv3 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # skip conv (if applicable)
        if (self.input_channels != self.output_channels) or any([i != j for i, j in zip(input_size, size_after_stride)]):
            assert isinstance(self.skip, nn.Sequential)
            output_size_skip = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        else:
            assert not isinstance(self.skip, nn.Sequential)
            output_size_skip = 0
        return output_size_conv1 + output_size_conv2 + output_size_conv3 + output_size_skip


class StackedMedNeXTBlocks(nn.Module):

    def __init__(self,
                 num_convs: int,
                 conv_op,
                 input_channels: int,
                 output_channels: ToIterableInt,
                 kernel_size: ToIterableInt,
                 initial_stride: int,
                 r: ToIterableInt,
                 norm_op,
                 norm_op_kwargs,
                 nonlin,
                 nonlin_kwargs = None,
                 first_conv_kernel_size: int = None,
                 hidden_dim: int = None,
                 residual: bool = True,
                 perform_second_nonlin: bool = True,
                 conv_bias: bool = False):

        super().__init__()

        self.initial_stride = initial_stride

        output_channels = maybe_convert_scalar_to_list(output_channels, num_convs)

        # for compat when we pass [3] * ndim instead of 3
        if isinstance(kernel_size, Iterable):
            assert len(kernel_size) == convert_conv_op_to_dim(conv_op)
            kernel_size = (kernel_size,) * num_convs
        kernel_size = maybe_convert_scalar_to_list(kernel_size, num_convs)
        r = maybe_convert_scalar_to_list(r, num_convs)

        # to reproduce the stem in the original impl
        first_conv_kernel_size = first_conv_kernel_size or kernel_size[0]

        # only first convolutional block can perform downsampling
        self.convs = nn.Sequential(
            MedNeXtBlock(
                conv_op, input_channels, output_channels[0],
                initial_stride=initial_stride, initial_kernel_size=first_conv_kernel_size, r=r[0], norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                residual=residual, hidden_dim=hidden_dim,
                perform_second_nonlin=perform_second_nonlin, conv_bias=conv_bias
            ),
            *[
                MedNeXtBlock(
                    conv_op, output_channels[i-1], output_channels[i],
                    initial_stride=1, initial_kernel_size=kernel_size[i], r=r[i], norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                    residual=residual, hidden_dim=hidden_dim,
                    perform_second_nonlin=perform_second_nonlin, conv_bias=conv_bias
                )
                for i in range(1, num_convs)
            ]
        )

        self.initial_stride = maybe_convert_scalar_to_list(initial_stride, conv_op)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(
            self,
            conv_op,
            in_channels: int,
            out_channels: int,
            exp_r: int = 4,
            kernel_size: int = 7,
            do_res: bool = False,
            norm_type: str = 'group',
            grn: bool = False
    ):

        super().__init__(conv_op, in_channels, out_channels, exp_r, kernel_size,
                        do_res=do_res, norm_type=norm_type, grn=grn)

        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv_op(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
            )

        self.conv1 = conv_op(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size // 2,
            groups = in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(
            self,
            conv_op,
            in_channels: int,
            out_channels: int,
            exp_r: int = 4,
            kernel_size: int = 7,
            do_res: bool = False,
            norm_type: str = 'group',
            grn: bool = False
    ):

        super().__init__(conv_op, in_channels, out_channels, exp_r, kernel_size,
                         do_res=do_res, norm_type = norm_type, grn=grn)

        self.resample_do_res = do_res

        self.padding_forward = (1, 0,) * convert_conv_op_to_dim(conv_op)

        if do_res:
            self.res_conv = conv_op(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
                )

        self.conv1 = conv_op(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size//2,
            groups = in_channels,
        )


    def forward(self, x, dummy_tensor = None):

        x1 = super().forward(x)
        # Asymmetry but necessary to match shape

        x1 = torch.nn.functional.pad(x1, self.padding_forward)

        if self.resample_do_res:
            res = self.res_conv(x)
            res = torch.nn.functional.pad(res, self.padding_forward)
            x1 += res

        return x1

class OutBlock(nn.Module):

    def __init__(self, conv_op, in_channels: int, n_classes: int):
        super().__init__()
        self.conv_out = conv_op(in_channels, n_classes, kernel_size=1)

    def forward(self, x, dummy_tensor=None):
        return self.conv_out(x)

class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(
            self,
            normalized_shape,
            eps = 1e-5,
            data_format = "channels_last"
    ):

        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))        # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))         # gamma
        self.eps = eps

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError('Invalid input data format: %s.' % data_format)
        self.data_format = data_format

        self.normalized_shape = (normalized_shape, )

    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

def test_network_initialization(kwargs, seed):
    torch.manual_seed(seed)
    net1 = StackedMedNeXTBlocks(**kwargs)

    torch.manual_seed(seed)
    net2 = StackedMedNeXTBlocks(**kwargs)
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


if __name__ == "__main__":


    # network = nnUNeXtBlock(in_channels=12, out_channels=12, do_res=False).cuda()

    # with torch.no_grad():
    #     print(network)
    #     x = torch.zeros((2, 12, 8, 8, 8)).cuda()
    #     print(network(x).shape)

    # network = DownsampleBlock(in_channels=12, out_channels=24, do_res=False)

    # with torch.no_grad():
    #     print(network)
    #     x = torch.zeros((2, 12, 128, 128, 128))
    #     print(network(x).shape)

    # network = MedNeXtBlock(conv_op=nn.Conv3d, in_channels=12, out_channels=12, do_res=True, grn=True, norm_type='group').to('mps')
    # network = LayerNorm(normalized_shape=12, data_format='channels_last').cuda()
    # network.eval()
    # with torch.no_grad():
    #     print(network)
    #     x = torch.randn((2, 12, 64, 64, 64)).to('mps')
    #     print(network(x).shape)


    data = torch.randn((2, 1, 128, 128, 128))
    block = MedNeXtBlock(nn.Conv3d, 1, 32, 1, 3, 4, nn.GroupNorm, None, None, None, nn.GELU, {}, True, hidden_dim=32, perform_second_nonlin=False, conv_bias=True)
    assert block(data).shape == (2, 32, 128, 128, 128)

    data = torch.randn((2, 32, 128, 128, 128))
    loss = nn.CrossEntropyLoss()
    stacked_convs = StackedMedNeXTBlocks(num_convs=2, conv_op=nn.Conv3d, input_channels = 32, output_channels = 64,
                                         kernel_size=3, initial_stride=2, r=4, norm_op=nn.GroupNorm, norm_op_kwargs=None,
                                         nonlin=nn.GELU, nonlin_kwargs={}, residual=True)
    # print(stacked_convs)
    # print(stacked_convs(data).shape)
    print(stacked_convs.compute_conv_feature_map_size(data.shape[2:]))

    out = stacked_convs(data)
    y = torch.randint_like(out, 0, 2)
    l = loss(out, y)
    l.backward()