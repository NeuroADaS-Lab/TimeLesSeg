import torch
from torch import nn

from timelesseg.typing import ToIterableInt
from timelesseg.models.utils import (
    maybe_convert_scalar_to_list,
    features_per_stage,
    convert_conv_op_to_dim
)
from timelesseg.models.MedNeXT.encoder import MedNeXTEncoder
from timelesseg.models.MedNeXT.decoder import MedNeXTDecoder

from timelesseg.models.weight_init import InitWeights_He, init_last_bn_before_add_to_0


class MedNeXT(nn.Module):
    def __init__(
            self,
            input_channels: int,
            n_stages: int,
            features_per_stage: ToIterableInt,
            expansion_ratio_per_stage: ToIterableInt,
            conv_op,
            kernel_sizes: ToIterableInt,
            strides: ToIterableInt,
            conv_bias: bool,
            n_conv_per_stage: ToIterableInt,
            num_classes: int,
            n_conv_per_stage_decoder: ToIterableInt,
            norm_op = None,
            norm_op_kwargs = None,
            nonlin = None,
            nonlin_kwargs = None,
            deep_supervision: bool = True
    ):

        super().__init__()

        n_conv_per_stage = maybe_convert_scalar_to_list(n_conv_per_stage, n_stages)
        n_conv_per_stage_decoder = maybe_convert_scalar_to_list(n_conv_per_stage_decoder, n_stages - 1)
        expansion_ratio_per_stage = maybe_convert_scalar_to_list(expansion_ratio_per_stage, 2 * n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, (
            "n_conv_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_conv_per_stage: {n_conv_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )

        self.encoder = MedNeXTEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            expansion_ratio_per_stage=expansion_ratio_per_stage[:n_stages],
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            conv_bias=conv_bias,
            return_skips=True
        )

        self.decoder = MedNeXTDecoder(
            self.encoder, num_classes, n_conv_per_stage_decoder,
            expansion_ratio_per_stage = expansion_ratio_per_stage[n_stages:],
            deep_supervision=deep_supervision, conv_bias=conv_bias
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )

    # see
    # https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/6d9f47ebc53dfbc679c6c68a93f4f92d34cb3766/dynamic_network_architectures/architectures/unet.py#L279
    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)

if __name__ == "__main__":
    data = torch.rand((2, 1, 128, 128, 128))

    model = MedNeXT(
        input_channels=1,
        n_stages=5,
        features_per_stage=features_per_stage(32, 5, 320),
        expansion_ratio_per_stage=2,
        conv_op=nn.Conv3d,
        kernel_sizes=[3] * 5,
        strides = [1] + [2] * 4,
        conv_bias=True,
        n_conv_per_stage=[2] * 5,
        n_conv_per_stage_decoder=[2] * 4,
        norm_op=nn.GroupNorm,
        nonlin=nn.GELU,
        nonlin_kwargs={},
        num_classes=2
    )
    print(model)

    if False:
        import hiddenlayer as hl

        g = hl.build_graph(model, data, transforms=None)
        g.save("network_architecture.pdf")
        del g

    loss = nn.CrossEntropyLoss()

    out = model(data)[0]
    y = torch.randint_like(out, 0, 2)
    l = loss(out, y)
    l.backward()
