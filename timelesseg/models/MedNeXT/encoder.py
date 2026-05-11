import numpy as np
import torch
from torch import nn

from timelesseg.typing import ToIterableInt
from timelesseg.models.utils import maybe_convert_scalar_to_list, features_per_stage
from timelesseg.models.MedNeXT.blocks import StackedMedNeXTBlocks


class MedNeXTEncoder(nn.Module):
    def __init__(
            self,
            input_channels: int,
            n_stages: int,
            features_per_stage: ToIterableInt,
            conv_op,
            kernel_sizes: ToIterableInt,
            strides: ToIterableInt,
            n_conv_per_stage: ToIterableInt,
            expansion_ratio_per_stage: ToIterableInt,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs = None,
            dropout_op = None,
            dropout_op_kwargs = None,
            residual: bool = True,
            stem: bool = True,
            conv_bias: bool = False,
            return_skips: bool = True
    ):

        super().__init__()

        kernel_sizes = maybe_convert_scalar_to_list(kernel_sizes, n_stages)
        features_per_stage = maybe_convert_scalar_to_list(features_per_stage, n_stages)
        n_conv_per_stage = maybe_convert_scalar_to_list(n_conv_per_stage, n_stages)
        strides = maybe_convert_scalar_to_list(strides, n_stages)
        expansion_ratio_per_stage = maybe_convert_scalar_to_list(expansion_ratio_per_stage, n_stages)

        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        assert len(expansion_ratio_per_stage) == n_stages

        if stem:
            stem_ops = []
            stem_kernel_size = kernel_sizes[0]
            padding = np.array(stem_kernel_size) // 2
            stem_ops.append(
                conv_op(input_channels, features_per_stage[0], kernel_size=kernel_sizes[0], stride=1, padding=padding, bias=conv_bias)
            )
            if norm_op is not None:
                this_norm_op_kwargs = norm_op_kwargs or {'num_channels': features_per_stage[0], 'num_groups': features_per_stage[0]}
                stem_ops.append(norm_op(**this_norm_op_kwargs))
            if nonlin is not None:
                stem_ops.append(nonlin(**nonlin_kwargs or {}))

            self.stem = nn.Sequential(*stem_ops)
            input_channels = features_per_stage[0]
            self._stem_output_channels = input_channels
            self._stem_stride = maybe_convert_scalar_to_list(1, conv_op)

        else:
            self.stem = None

        stages = []
        for s in range(n_stages):
            # first convolutional layer which downsamples input after first stage
            conv_stride = strides[s]

            stages.append(
                StackedMedNeXTBlocks(
                    n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s],
                    kernel_size=kernel_sizes[s], initial_stride=conv_stride, r=expansion_ratio_per_stage[s],
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                    residual=residual, conv_bias=conv_bias
                )
            )

            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.n_stages = n_stages
        self.output_channels = features_per_stage
        self.return_skips = return_skips
        self.strides = [maybe_convert_scalar_to_list(i, conv_op) for i in strides]

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.kernel_sizes = kernel_sizes
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if not self.return_skips:
            return ret[-1]
        return ret

    def _compute_stem_feature_map_size(self, input_size):
        size_after_stride = [i // s for i, s in zip(input_size, self._stem_stride)]
        output = np.prod([self._stem_output_channels, *size_after_stride])
        return output

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self._compute_stem_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output

if __name__ == "__main__":
    input_channels = 1
    n_stages = 4
    start_features = 32
    ff = features_per_stage(start_features, n_stages, 320)

    enc = MedNeXTEncoder(
        input_channels,
        n_stages,
        ff,
        nn.Conv3d,
        3,
        strides = [1] + [2] * (n_stages - 1),
        n_conv_per_stage=[3] * n_stages,
        expansion_ratio_per_stage = [4] * n_stages,
        conv_bias=True,
        norm_op=nn.GroupNorm,
        norm_op_kwargs=None,
        nonlin=nn.GELU,
        nonlin_kwargs={},
        return_skips=False
    ).to('mps')

    print(enc)
    x = torch.randn((2, 1,) + (128,) * 3).to('mps')
    print(enc(x).shape)
    print(enc.compute_conv_feature_map_size(x.shape[2:]))