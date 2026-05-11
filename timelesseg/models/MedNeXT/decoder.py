import numpy as np
import torch
from torch import nn

from timelesseg.typing import ToIterableInt
from timelesseg.models.utils import (
    maybe_convert_scalar_to_list,
    get_matching_convtransp,
    features_per_stage
)
from timelesseg.models.MedNeXT.blocks import StackedMedNeXTBlocks
from timelesseg.models.MedNeXT.encoder import MedNeXTEncoder


class MedNeXTDecoder(nn.Module):
    def __init__(
            self,
            encoder: MedNeXTEncoder,
            num_classes: int,
            n_conv_per_stage: ToIterableInt,
            expansion_ratio_per_stage: ToIterableInt,
            deep_supervision,
            norm_op = None,
            norm_op_kwargs: dict = None,
            nonlin = None,
            nonlin_kwargs: dict = None,
            residual: bool = True,
            conv_bias: bool = None
    ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = encoder.n_stages

        n_conv_per_stage = maybe_convert_scalar_to_list(n_conv_per_stage, (n_stages_encoder - 1))

        assert len(n_conv_per_stage) == (n_stages_encoder - 1), "n_conv_per_stage must have as many entries as we have " \
                                                                "resolution stages - 1 (n_stages in encoder - 1), " \
                                                                "here: %d" % n_stages_encoder
        assert len(expansion_ratio_per_stage) == (n_stages_encoder - 1)

        transpconv_op = get_matching_convtransp(encoder.conv_op)
        conv_bias = conv_bias or encoder.conv_bias
        norm_op = norm_op or encoder.norm_op
        norm_op_kwargs = norm_op_kwargs or encoder.norm_op_kwargs
        # dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        # dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = nonlin or encoder.nonlin
        nonlin_kwargs = nonlin_kwargs or encoder.nonlin_kwargs

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedMedNeXTBlocks(
                num_convs=n_conv_per_stage[s - 1],
                conv_op=encoder.conv_op,
                input_channels=2 * input_features_skip,
                output_channels=input_features_skip,
                kernel_size=encoder.kernel_sizes[-(s + 1)],
                initial_stride=1,
                r=expansion_ratio_per_stage[s - 1],
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                conv_bias=conv_bias,
                residual = residual
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output

if __name__ == "__main__":
    input_channels = 1
    num_stages = 5
    start_features = 32
    ff = features_per_stage(start_features, num_stages, 320)
    enc = MedNeXTEncoder(
        input_channels=input_channels,
        n_stages=num_stages,
        features_per_stage=ff,
        conv_op=nn.Conv3d,
        kernel_sizes = 3,
        strides = [1] + [2] * (num_stages - 1),
        n_conv_per_stage=[3] * num_stages,
        expansion_ratio_per_stage = [2] * num_stages,
        norm_op=nn.GroupNorm,
        norm_op_kwargs=None,
        nonlin=nn.GELU,
        nonlin_kwargs={},
        return_skips=True
    ).to('mps')

    dec = MedNeXTDecoder(
        enc,
        num_classes=2,
        n_conv_per_stage = [3] * (num_stages - 1),
        expansion_ratio_per_stage = [2] * (num_stages - 1),
        deep_supervision=True,
        norm_op=None,
        norm_op_kwargs=None
    ).to('mps')

    # print(dec)

    x = torch.randn((2, 1,) + (128,) * 3).to('mps')
    skips = enc(x)
    print(f'Shape after bottleneck: {skips[-1].shape}')
    print(dec(skips)[0].shape)
    print(dec.compute_conv_feature_map_size(x.shape[2:]))